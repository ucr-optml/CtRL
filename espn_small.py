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
                    default='./checkpoints/{mode}/s{seed_data}-{arch}-{dataset}/runseed-{seed}',
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
parser.add_argument('--alloc_mode', type=str, default='default')
parser.add_argument('--sep_header', dest='sep_header', default=False, action='store_true')


def main():
    args = parser.parse_args()
    if args.taskIDs == []:
        args.taskIDs = [i for i in range(args.num_tasks)]
    print('args = ', args)

    # Prepare
    if not torch.cuda.is_available():
        ValueError('no gpu device available')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # Model
    if args.arch in ['FC1024']:
        model = models.__dict__[args.arch](num_classes=10)
    elif args.arch in ['LeNet']:
        model = models.__dict__[args.arch](num_classes=10)
    elif args.arch in ['LeNet5']:
        model = models.__dict__[args.arch](num_classes=10)
    else:
        ValueError('TODO: model {}'.format(args.arch))
    model = nn.DataParallel(model)
    model = model.module.cuda()
    print(model)

    # 
    filepath = args.checkpoint_format.format(seed=args.seed, arch=args.arch, dataset=args.dataset, seed_data=args.seed_data, mode=args.alloc_mode)
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath+'/shared_info.pth.tar')
        trained_tasks = checkpoint['trained_tasks']
        state_dict = checkpoint['state_dict']
        free_masks = checkpoint['free_masks']
        for n, m in model.named_modules():
            if isinstance(m, Cell):
                m.free_conv_mask = free_masks[n]
        if not args.sep_header:
            model.__setattr__('free_last_mask', free_masks['last'])
        model.load_state_dict(state_dict)
    else:
        os.makedirs(filepath)
        checkpoint = {}
        trained_tasks = []
        free_masks = {}
        for n, m in model.named_modules():
            if isinstance(m, Cell):
                free_masks[n] = torch.ones(m.conv.weight.shape, dtype=torch.bool).cuda()
        if not args.sep_header:
            free_masks['last'] = torch.ones(model.last.weight.shape, dtype=torch.bool).cuda()
            model.__setattr__('free_last_mask', free_masks['last'])
        checkpoint['trained_tasks'] = trained_tasks
        checkpoint['state_dict'] = model.state_dict()
        checkpoint['free_masks'] = free_masks
        torch.save(checkpoint, filepath+'/shared_info.pth.tar')
        
    if 'PermutedMNIST' in args.dataset:
        data_loaders = PermutedMNIST(args)
    elif 'RotatedMNIST' in args.dataset:
        data_loaders = RotatedMNIST(args)
    else:
        ValueError('TODO: dataset {}'.format(args.dataset))
    manager = Manager(args, model, data_loaders, filepath)
    trained_num=len(trained_tasks)
    accuracy = []
    for i in range(len(args.taskIDs)):
        task_id = args.taskIDs[i]
        if task_id in trained_tasks:
            print('Task {} is already trained. Evaluating...'.format(task_id))
            summary = manager.validate(task_id)
            accuracy.append(summary['acc'])
            continue
        print('Train task {} ...'.format(task_id))
        manager.add_task(task_id)
        manager.warmup(args.warmup)
        for epoch in range(args.warmup, args.pruning_iter[0]):
            manager.train(epoch)
        # get iterative pruning ratio
        res_FLOP_iter = manager.get_res_FLOP_iter()
        res_sparsity_iter = manager.get_res_sparsity_iter(trained_num)
        print('res_FLOP_iter: ', res_FLOP_iter)
        print('res_sparsity_iter: ', res_sparsity_iter)
        for epoch in range(args.pruning_iter[0], args.pruning_iter[1]):
            manager.prune(res_FLOP_iter[epoch], res_sparsity_iter[epoch])
            manager.train(epoch, 'prune')
        for m in model.modules():
            if isinstance(m, Cell):
                # m.value = nn.Parameter(torch.ones(m.value.shape, dtype=torch.float)).cuda()
                nn.init.ones_(m.value)
                m.value.requires_grad_(False)
        for epoch in range(args.pruning_iter[1], args.epochs):
            manager.train(epoch, 'finetune')
        manager.save_checkpoint(task_id)
        manager.update_free_conv_mask()
        trained_num += 1

        # save checkpoint for main body
        free_masks = {}
        for n, m in model.named_modules():
            if isinstance(m, Cell):
                free_masks[n] = m.free_conv_mask
        if not args.sep_header:
            free_masks['last'] = model.free_last_mask
        trained_tasks.append(task_id)
       
        checkpoint = {
            'state_dict': model.state_dict(),
            'free_masks': free_masks,
            'trained_tasks': trained_tasks,
        }
        torch.save(checkpoint, filepath+'/shared_info.pth.tar')
        for j in range(i+1):
            summary = manager.validate(args.taskIDs[j])
        accuracy.append(summary['acc'])
        # print(manager.compute_sparsity('fixed'))

    print(accuracy)
    print(sum(accuracy)/len(accuracy))
    # print(manager.compute_sparsity('fixed'))

if __name__ == '__main__':
    main()