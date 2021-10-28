import os
import argparse

import torch
import torch.nn as nn

import models_small as models
from models_small import Cell
from utils.manager_small import Manager
from dataloader import PermutedMNIST, RotatedMNIST


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_format', type=str,
                    default='./checkpoints/s{seed_data}-{arch}-{dataset}/runseed{seed}',
                    help='checkpoint file format')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed')
parser.add_argument('--seed_data', type=int, default=1,
                    help='Random seed to generate random tasks')
parser.add_argument('--arch', type=str, default='FC1024', 
                    help='Architecture')
parser.add_argument('--dataset', type=str, default='RotatedMNIST', 
                    help='Dataset')
parser.add_argument('--num_tasks', type=int, default=36)
parser.add_argument('--data_dir', type=str, default='data', 
                    help='Data directory')
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', nargs="+", type=int, default=10)
parser.add_argument('--warmup', type=int, default=1)
parser.add_argument('--pruning_iter', nargs="+", type=int, default=[3, 7])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--optimizer', type=str, default='RMSprop')
parser.add_argument('--reg', type=str, default='flop_0.5')
parser.add_argument('--lam', type=float, default=1)
parser.add_argument('--rho', type=float, default=0.05)
parser.add_argument('--res_FLOP', type=float, default=0.2)
parser.add_argument('--taskIDs', nargs="+", type=int, 
                    default=[],
                    help='The list of taskID to infer')

parser.add_argument('--ratio_samples', nargs="+", type=float, default=[])


def main():
    args = parser.parse_args()
    if args.taskIDs == []:
        args.taskIDs = [i for i in range(args.num_tasks)]
    print('args = ', args)

    # Prepare
    if not torch.cuda.is_available():
        raise ValueError('no gpu device available')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # Model
    if args.arch in ['FC1024']:
        model = models.__dict__[args.arch](num_classes=10)
    else:
        raise ValueError('TODO: model {}'.format(args.arch))
    model = nn.DataParallel(model)
    model = model.module.cuda()
    print(model)

    # 
    filepath = args.checkpoint_format.format(seed=args.seed, arch=args.arch, \
        dataset=args.dataset, seed_data=args.seed_data)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    if os.path.exists(filepath+'/shared_info.pth.tar'):
        model.load_state_dict(filepath+'/shared_info.pth.tar')
    else:
        model.save_state_dict(filepath+'/shared_info.pth.tar')
        
    if 'PermutedMNIST' in args.dataset:
        data_loaders = PermutedMNIST(args)
    elif 'RotatedMNIST' in args.dataset:
        data_loaders = RotatedMNIST(args)
    else:
        raise ValueError('TODO: dataset {}'.format(args.dataset))
    manager = Manager(args, model, data_loaders, filepath)
    accuracy = []
    for i in range(len(args.taskIDs)):
        task_id = args.taskIDs[i]
        if model.task_exists(task_id):
            print('Task {} is already trained. Evaluating...'.format(task_id))
            # manager.set_task(task_id)
            summary = manager.validate(task_id)
            accuracy.append(summary['acc'])
            continue
        print('Train task {} ...'.format(task_id))
        manager.add_task(task_id)
        # warmup and pretrain
        manager.warmup(args.warmup)
        for epoch in range(args.warmup, args.pruning_iter[0]):
            manager.train(epoch)
        # get iterative pruning ratio
        res_FLOP_iter = manager.get_res_FLOP_iter()
        res_sparsity_iter = manager.get_res_sparsity_iter()
        print('res_FLOP_iter: ', res_FLOP_iter)
        print('res_sparsity_iter: ', res_sparsity_iter)
        # pruning
        for epoch in range(args.pruning_iter[0], args.pruning_iter[1]):
            manager.prune(res_FLOP_iter[epoch], res_sparsity_iter[epoch])
            manager.train(epoch, mode='prune')
        # finetuning
        for m in model.modules():
            if isinstance(m, Cell):
                # m.value = nn.Parameter(torch.ones(m.value.shape, dtype=torch.float)).cuda()
                nn.init.ones_(m.value)
                m.value.requires_grad_(False)
        for epoch in range(args.pruning_iter[1], args.epochs):
            manager.train(epoch, mode='finetune')
        manager.save_checkpoint(task_id)
        manager.update_free_mask()

        # save checkpoint for main body
        model.save_state_dict(filepath+'/shared_info.pth.tar')
        for j in range(i+1):
            summary = manager.validate(args.taskIDs[j])
        accuracy.append(summary['acc'])
        # print(manager.compute_sparsity('fixed'))

    print(accuracy)
    print(sum(accuracy)/len(accuracy))
    # print(manager.compute_sparsity('fixed'))

if __name__ == '__main__':
    main()