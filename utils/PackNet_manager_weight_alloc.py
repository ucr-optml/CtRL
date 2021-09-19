from re import S
import math
from torch._C import TracingState
from torchvision import transforms
from models.cell import Cell
import torch
import torch.nn as nn
from torch.nn import Conv2d, Linear, BatchNorm2d
from tqdm import tqdm
from . import Metric, classification_accuracy
# from . import Pruner
import logging

class Manager(object):

    
    def __init__(self, args, model, data_loaders, filepath) -> None:
        super().__init__()
        self.args = args
        self.model = model
        self.data_loaders = data_loaders
        self.filepath = filepath

        self.init_optimizer()
        self.criterion = nn.CrossEntropyLoss()


        self.shared_weights = {}
        for n, m in self.model.named_modules():
            if isinstance(m, Conv2d):
                self.shared_weights[n] = m.weight.clone()

        self.masks = {}
        for n, m in self.model.named_modules():
            if isinstance(m, (Conv2d, Linear)):
                self.masks[n] = torch.ones(m.weight.shape, dtype=torch.bool).cuda()


    def update_free_conv_mask(self):
        for n, m in self.model.named_modules():
            if isinstance(m, Conv2d):
                self.shared_weights[n] = self.shared_weights[n] * ~self.model.free_conv_masks[n] + m.weight * self.model.free_conv_masks[n]
                self.model.free_conv_masks[n] = self.model.free_conv_masks[n] & ~self.masks[n]

    def init_optimizer(self):
        optimizer_arg = {'params':self.model.parameters(),
                         'lr':self.args.lr}
        self.optimizer = torch.optim.__dict__[self.args.optimizer](**optimizer_arg)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.schedule, gamma=0.1)

    def init_weights(self):
        with torch.no_grad():
            for n, m in self.model.named_modules():
                if isinstance(m, BatchNorm2d):
                    if self.args.bn:
                        nn.init.ones_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    m.running_mean.zero_()
                    m.running_var.fill_(1)
                    m.num_batches_tracked.zero_()
                if isinstance(m, Conv2d):
                    weight = m.weight.clone()
                    m.weight = nn.Parameter(weight*~self.model.free_conv_masks[n] + nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu').cuda() * self.model.free_conv_masks[n])
                if isinstance(m, Linear):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, inputs, targets):
        out = self.model.forward(inputs)
        loss = self.criterion(out, targets)

        weights = torch.cat([m.weight.view(-1) for m in self.model.modules() if isinstance(m, Conv2d)])
        l2_loss = 1./torch.sqrt(torch.tensor(weights.numel())) * torch.norm(weights, 2)
        loss += self.args.lam * l2_loss

        self.optimizer.zero_grad()
        loss.backward()
        return loss.detach(), out
    
    def fixed_weight_pruning(self, ratio):
        for n, m in self.model.named_modules():
            if isinstance(m, Conv2d) and int(ratio * m.weight.numel()) > 0:
                thred = torch.topk((m.weight.abs() * ~self.model.free_conv_masks[n]).view(-1), int(ratio * m.weight.numel()))[0][-1]
                self.masks[n] = self.model.free_conv_masks[n] | ((m.weight.abs() >= thred) & ~self.model.free_conv_masks[n])
        with torch.no_grad():
            for n, m in self.model.named_modules():
                if isinstance(m, (Conv2d, Linear)):
                    m.weight *= self.masks[n]

    def get_res_sparsity_iter(self, trained_num, idx=-1):
        upper = self.args.rho
        for _ in range(trained_num):
            upper = upper * (1 - self.args.rho) + self.args.rho
        if idx >= 0:
            upper = self.args.res_sparsity[idx]
        res_sparsity_iter = {}
        print('sparsity before pruning: {}, upper: {}'.format(compute_sparsity(self.masks), upper))
        for e in range(self.args.pruning_iter[0], self.args.pruning_iter[1]):
            res_sparsity_iter[e] = compute_sparsity(self.masks) - (e + 1 - self.args.pruning_iter[0])*(compute_sparsity(self.masks)-upper)/(self.args.pruning_iter[1]-self.args.pruning_iter[0])

        return res_sparsity_iter

    def prune(self, res_sparsity):
        # sparsity res
        
        for n, m in self.model.named_modules():
            if isinstance(m, Conv2d):
                thred = torch.topk((m.weight * self.model.free_conv_masks[n]).abs().view(-1), max(int(m.weight.numel() * res_sparsity - ((~self.model.free_conv_masks[n]) & self.masks[n]).sum()),1))[0][-1]
                self.masks[n] = (~self.model.free_conv_masks[n] & self.masks[n]) | ((m.weight.abs() >= thred) & self.model.free_conv_masks[n])
            if isinstance(m, Linear):
                thred = torch.topk(m.weight.abs().view(-1), max(int(m.weight.numel() * res_sparsity),1))[0][-1]
                self.masks[n] = m.weight.abs() >= thred   
        with torch.no_grad():
            for n, m in self.model.named_modules():
                if isinstance(m, (Conv2d, Linear)):
                    m.weight *= self.masks[n]
        print(compute_sparsity(self.masks), res_sparsity)

    def train(self, epoch_idx, mode=None):
        self.model.train()
        train_loader = self.data_loaders.train_loader
        train_loss = Metric('train_loss')
        train_accuracy = Metric('train_accuracy')

        with tqdm(total=len(train_loader),
                  desc='Train Ep. #{}: '.format(epoch_idx + 1),
                  disable=False,
                  ascii=True) as t:
            for batch_idx, (input, target) in enumerate(train_loader):
                input, target = input.cuda(), target.cuda()
                loss, output = self.forward(input, target)
        
                
                # update grads
                for n, m in self.model.named_modules():
                    if isinstance(m, Conv2d):
                        m.weight.grad *= (self.model.free_conv_masks[n] & self.masks[n])
                    if isinstance(m, Linear):
                        m.weight.grad *= self.masks[n]
                
                self.optimizer.step()
        
                num = input.size(0)
                train_accuracy.update(classification_accuracy(output, target), num)
                train_loss.update(loss.cpu(), num)


                t.set_postfix({'loss': train_loss.avg.item(),
                               'accuracy': '{:.2f}'.format(100. * train_accuracy.avg.item()),
                               # 'lr': self.optimizer.param_groups['lr'],
                               # 'sparsity': self.pruner.calculate_sparsity()
                               })
                t.update(1)
            self.scheduler.step()
            summary = {'loss': '{:.3f}'.format(train_loss.avg.item()),
                       'accuracy': '{:.2f}'.format(100. * train_accuracy.avg.item())}
        return summary

    def validate(self, task_id=None):
        self.model.eval()
        if task_id is not None:
            self.apply_task(task_id)
            self.data_loaders.update_task(task_id)
        val_loader = self.data_loaders.val_loader

        loss = Metric('loss')
        accuracy = Metric('accuracy')

        with tqdm(total=len(val_loader),
                  desc='Val: ',
                  ascii=True) as t:
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.cuda(), target.cuda()

                    output = self.model(data)
                    num = data.size(0)
                    loss.update(self.criterion(output, target).cpu(), num)
                    accuracy.update(classification_accuracy(output, target), num)

                    t.set_postfix({ 'loss': loss.avg.item(),
                                    'accuracy': '{:.2f}'.format(100. * accuracy.avg.item()),
                                    # 'sparsity': self.pruner.calculate_sparsity(),
                                    # 'task{} ratio'.format(self.inference_dataset_idx): self.pruner.calculate_curr_task_ratio(),
                                    # 'zero ratio': self.pruner.calculate_zero_ratio(),
                                    # 'mpl': self.args.network_width_multiplier,
                                    # 'shared_ratio': self.pruner.calculate_shared_part_ratio(),
                                    })
                    t.update(1)

        summary = {'loss': '{:.3f}'.format(loss.avg.item()),
                   'accuracy': '{:.2f}'.format(100. * accuracy.avg.item()),
                   # 'sparsity': '{:.3f}'.format(self.pruner.calculate_sparsity()),
                   # 'task{} ratio'.format(self.inference_dataset_idx): '{:.3f}'.format(self.pruner.calculate_curr_task_ratio()),
                   # 'zero ratio': '{:.3f}'.format(self.pruner.calculate_zero_ratio()),
                   # 'mpl': self.args.network_width_multiplier,
                   # 'shared_ratio' : '{:.3f}'.format(self.pruner.calculate_shared_part_ratio())
                   }

        # if self.args.log_path:
        #     logging.info(('In validate()-> Val Ep. #{} '.format(epoch_idx + 1)
        #                  + ', '.join(['{}: {}'.format(k, v) for k, v in summary.items()])))
        # return accuracy.avg.item()
        return summary
    
    def apply_task(self, task_id):
        self.data_loaders.update_task(task_id)
        self.model.eval()
        checkpoint = torch.load(self.filepath+'/{}.pth.tar'.format(task_id))
        channel_weights = checkpoint['channel_weights']
        channel_biases = checkpoint['channel_biases']
        running_means = checkpoint['running_means']
        running_vars = checkpoint['running_vars']
        num_batches_tracked = checkpoint['num_batches_tracked']
        masks = checkpoint['masks']
        header_weight = checkpoint['header_weight']
        header_bias = checkpoint['header_bias']

        with torch.no_grad():
            for n, m in self.model.named_modules():
                if isinstance(m, BatchNorm2d):
                    if self.args.bn:
                        m.weight = torch.nn.Parameter(channel_weights[n]).cuda()
                        if m.bias is not None:
                            m.bias = torch.nn.Parameter(channel_biases[n]).cuda()
                    m.running_mean = running_means[n].cuda()
                    m.running_var = running_vars[n].cuda()
                    m.num_batches_tracked = num_batches_tracked[n].cuda()
                if isinstance(m, Conv2d):
                    m.weight = nn.Parameter(self.shared_weights[n] * masks[n]).cuda()
                if isinstance(m, Linear):
                    m.weight = nn.Parameter(header_weight * masks[n]).cuda()
                    if m.bias is not None:
                        m.bias = nn.Parameter(header_bias).cuda()    

    def add_task(self, task_id):
        self.data_loaders.update_task(task_id)
        self.model.eval()
        self.init_weights()
        self.masks = {}
        with torch.no_grad():
            for n, m in self.model.named_modules():
                if isinstance(m, (Conv2d, Linear)):
                    self.masks[n] = torch.ones(m.weight.shape, dtype=torch.bool).cuda()


    def save_checkpoint(self, task_id):
        channel_weights = {}
        channel_biases = {}
        running_means = {}
        running_vars = {}
        num_batches_tracked = {}
        masks = self.masks
        header_weight = None
        header_bias = None
        for n, m in self.model.named_modules():
            if isinstance(m, BatchNorm2d):
                if self.args.bn:
                    channel_weights[n] = m.weight
                    if m.bias is not None:
                        channel_biases[n] = m.bias
                running_means[n] = m.running_mean
                running_vars[n] = m.running_var
                num_batches_tracked[n] = m.num_batches_tracked
            if isinstance(m, Linear):
                header_weight = m.weight
                if m.bias is not None:
                    header_bias = m.bias
        checkpoint = {
            'channel_weights': channel_weights,
            'channel_biases': channel_biases,
            'running_means': running_means,
            'running_vars': running_vars,
            'num_batches_tracked': num_batches_tracked,
            'masks': masks,
            'header_weight': header_weight,
            'header_bias': header_bias,
        }
        torch.save(checkpoint, self.filepath + '/{}.pth.tar'.format(task_id))

        
                    

def compute_sparsity(masks) -> float:

    sum = 0.
    count = 0.
    for m in masks.values():
        sum += m.sum()
        count += m.numel()
    return sum / count