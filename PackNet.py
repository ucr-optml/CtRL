import os
import argparse

import torch
import torch.nn as nn
from torch.nn import Conv2d

import PackNet_models as models
from utils.PackNet_manager import Manager
from dataloader import RandSplitCIFAR100, RandSplitImageNet


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_format', type=str,
                    default='./checkpoints_PackNet/{mode}/s{seed_data}-{arch}-{dataset}/runseed-{seed}',
                    help='checkpoint file format')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed')
parser.add_argument('--seed_data', type=int, default=1,
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
parser.add_argument('--lam', type=float, default=1)
parser.add_argument('--rho', type=float, default=0.1)
parser.add_argument('--taskIDs', nargs="+", type=int, default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
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
        model = models.__dict__[args.arch](num_classes=args.num_classes, affine=False)
    else:
        ValueError('no valid arch')
    model = nn.DataParallel(model)
    model = model.module.cuda()
    print(model)

    # 
    filepath = args.checkpoint_format.format(seed=args.seed, arch=args.arch, \
        dataset=args.dataset, seed_data=args.seed_data, mode=args.alloc_mode)
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath+'/shared_info.pth.tar')
        trained_tasks = checkpoint['trained_tasks']
        state_dict = checkpoint['state_dict']
        free_masks = checkpoint['free_masks']
        for n, m in model.named_modules():
            if isinstance(m, Conv2d):
                m.free_conv_mask = free_masks[n]
        model.load_state_dict(state_dict)
    else:
        os.makedirs(filepath)
        checkpoint = {}
        trained_tasks = []
        free_masks = {}
        for n, m in model.named_modules():
            if isinstance(m, Conv2d):
                m.free_conv_mask = torch.ones(m.weight.shape, dtype=torch.bool).cuda()
                free_masks[n] = m.free_conv_mask
        checkpoint['trained_tasks'] = trained_tasks
        checkpoint['state_dict'] = model.state_dict()
        checkpoint['free_masks'] = free_masks
        torch.save(checkpoint, filepath+'/shared_info.pth.tar')

        
    if 'SplitCIFAR100' in args.dataset:
        data_loaders = RandSplitCIFAR100(args)
    elif 'SplitImageNet' in args.dataset:
        data_loaders = RandSplitImageNet(args)
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
        res_sparsity_iter = manager.get_res_sparsity_iter(trained_num)
        print('res_sparsity_iter: ', res_sparsity_iter)
        for epoch in range(args.pruning_iter[0], args.pruning_iter[1]):
            manager.prune(res_sparsity_iter[epoch])
            manager.train(epoch, 'prune')
        for epoch in range(args.pruning_iter[1], args.epochs):
            manager.train(epoch, 'finetune')
        manager.save_checkpoint(task_id)
        manager.update_free_conv_mask()
        trained_num += 1

        # save checkpoint for main body
        free_masks = {}
        for n, m in model.named_modules():
            if isinstance(m, Conv2d):
                free_masks[n] = m.free_conv_mask
        trained_tasks.append(task_id)

        checkpoint = {
            'state_dict': model.state_dict(),
            'free_masks': free_masks,
            'trained_tasks': trained_tasks
        }
        torch.save(checkpoint, filepath+'/shared_info.pth.tar')
        for j in range(i+1):
            summary = manager.validate(args.taskIDs[j])
        accuracy.append(summary['acc'])

    print(accuracy)
    print(sum(accuracy)/len(accuracy))

if __name__ == '__main__':
    main()