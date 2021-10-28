import os
import argparse

import torch
import torch.nn as nn

import models
from models import Cell
from utils.manager import Manager
from dataloader import RandSplitCIFAR100, RandSplitImageNet, CIFAR10


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_format', type=str,
                    default='./checkpoints/{mode}/s{seed_data}-{arch}-{dataset}/runseed-{seed}',
                    help='checkpoint file format')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed')
parser.add_argument('--seed_data', type=int, default=-1,
                    help='Random seed to generate random tasks')
parser.add_argument('--arch', type=str, default='GEMResNet18', 
                    help='Architecture')
parser.add_argument('--dataset', type=str, default='SplitCIFAR100', 
                    help='Dataset')
parser.add_argument('--num_classes', type=int, default=5, 
                    help='number of classes')
parser.add_argument('--data_dir', type=str, default='data', 
                    help='Data directory')
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--warmup', type=int, default=10)
parser.add_argument('--pruning_iter', nargs="+", type=int, default=[60, 150])
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--reg', type=str, default='flop_0.5')
parser.add_argument('--lam', type=float, default=1)
parser.add_argument('--rho', type=float, default=0.1)
parser.add_argument('--res_FLOP', type=float, default=0.2)
parser.add_argument('--taskIDs', nargs="+", type=int, 
                    default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
                    help='The list of taskID to infer')

parser.add_argument('--ratio_samples', nargs="+", type=float, default=[])
parser.add_argument('--alloc_mode', type=str, default='default')



def main():
    args = parser.parse_args()
    print('args = ', args)

    # Prepare
    if not torch.cuda.is_available():
        ValueError('no gpu device available')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # Model
    if args.arch in ['GEMResNet18', 'ResNet50']:
        model = models.__dict__[args.arch]()
    else:
        raise ValueError('TODO: model {}'.format(args.arch))
    model = nn.DataParallel(model)
    model = model.module.cuda()
    # print(model)

    # 
    filepath = args.checkpoint_format.format(seed=args.seed, arch=args.arch, \
        dataset=args.dataset, seed_data=args.seed_data, mode=args.alloc_mode)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    if os.path.exists(filepath+'/shared_info.pth.tar'):
        model.load_state_dict(filepath+'/shared_info.pth.tar')
    else:
        model.save_state_dict(filepath+'/shared_info.pth.tar')
        
    if 'SplitCIFAR100' in args.dataset:
        data_loaders = RandSplitCIFAR100(args)
    elif 'SplitImageNet' in args.dataset:
        data_loaders = RandSplitImageNet(args)
    elif 'CIFAR10' in args.dataset:
        data_loaders = CIFAR10(args)
    else:
        raise ValueError('TODO: dataset {}'.format(args.dataset))
    manager = Manager(args, model, data_loaders, filepath)
    accuracy = []
    channel = []
    for i in range(len(args.taskIDs)):
        task_id = args.taskIDs[i]
        if model.task_exists(task_id):
            print('Task {} is already trained. Evaluating...'.format(task_id))
            manager.set_task(task_id)
            summary = manager.validate(task_id)
            accuracy.append(summary['acc'])
            channel.append(channel_count(model))
            continue
        print('Train task {} ...'.format(task_id))
        manager.add_task(task_id, num_classes=args.num_classes)
        # warmup and pretrain
        manager.warmup(args.warmup, task_id)
        for epoch in range(args.warmup, args.pruning_iter[0]):
            manager.train(epoch, task_id)
        # get iterative pruning ratio
        res_FLOP_iter = manager.get_res_FLOP_iter()
        res_sparsity_iter = manager.get_res_sparsity_iter()
        print('res_FLOP_iter: ', res_FLOP_iter)
        print('res_sparsity_iter: ', res_sparsity_iter)
        # pruning
        for epoch in range(args.pruning_iter[0], args.pruning_iter[1]):
            manager.prune(task_id, res_FLOP_iter[epoch], res_sparsity_iter[epoch])
            manager.train(epoch, task_id, mode='prune')
        # finetuning
        for epoch in range(args.pruning_iter[1], args.epochs):
            manager.train(epoch, task_id, mode='finetune')
        manager.save_checkpoint(task_id)
        manager.update_free_conv_mask()

        # save checkpoint for main body
        model.save_state_dict(filepath+'/shared_info.pth.tar')
        for j in range(i+1):
            manager.set_task(args.taskIDs[j])
            summary = manager.validate(args.taskIDs[j])
        accuracy.append(summary['acc'])

    print(accuracy)
    print(sum(accuracy)/len(accuracy))
    print(channel)

def channel_count(model):
    all = 0
    num = 0
    for m in model.modules():
        if isinstance(m, Cell):
            all += m.get_output_mask().numel()
            num += m.get_output_mask().sum()
    return (num.float()/all).item()

if __name__ == '__main__':
    main()