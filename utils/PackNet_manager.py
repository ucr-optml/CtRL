import math
import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d
from tqdm import tqdm
from . import Metric, classification_accuracy

class Manager(object):

    
    def __init__(self, args, model, data_loaders, filepath) -> None:
        super().__init__()
        self.args = args
        self.model = model
        self.data_loaders = data_loaders
        self.filepath = filepath

        self.init_optimizer()
        self.criterion = nn.CrossEntropyLoss()

        self.sparsity = 1.

        for m in self.model.modules():
            if isinstance(m, Conv2d):
                m.mask = torch.ones(m.weight.shape, dtype=torch.bool).cuda()

    def update_free_conv_mask(self):
        for m in self.model.modules():
            if isinstance(m, Conv2d):
                m.free_conv_mask = (m.free_conv_mask & ~m.mask)

    def init_optimizer(self):
        optimizer_arg = {'params':self.model.parameters(),
                         'lr':self.args.lr}
        if 'Adam' in self.args.optimizer:
            optimizer_arg['betas'] = (0, 0.999)
        self.optimizer = torch.optim.__dict__[self.args.optimizer](**optimizer_arg)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.epochs - self.args.pruning_iter[-1])

    def init_weights(self):
        with torch.no_grad():
            for m in self.model.modules():
                if isinstance(m, BatchNorm2d):
                    m.running_mean.zero_()
                    m.running_var.fill_(1)
                    m.num_batches_tracked.zero_()
                if isinstance(m, Conv2d):
                    m.weight.data = m.weight.data*~m.free_conv_mask + nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu').cuda() * m.free_conv_mask
            nn.init.kaiming_uniform_(self.model.last.weight, a=math.sqrt(5))
            if self.model.last.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.model.last.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.model.last.bias, -bound, bound)

    def forward(self, inputs, targets):
        out = self.model.forward(inputs)
        loss = self.criterion(out, targets)

        weights = torch.cat([(m.weight * m.mask).view(-1) for m in self.model.modules() if isinstance(m, Conv2d)])
        num = torch.cat([m.mask.view(-1) for m in self.model.modules() if isinstance(m, Conv2d)]).sum()
        l2_loss = 1./math.sqrt(num) * torch.norm(weights, 2)
        loss += self.args.lam * l2_loss

        self.optimizer.zero_grad()
        loss.backward()
        return loss.detach(), out
    
    def get_res_sparsity_iter(self, trained_num):
        if self.args.alloc_mode == 'default':
            upper = self.args.rho
        elif self.args.alloc_mode == 'exp-inf':
            upper = 0
            for _ in range(trained_num + 1):
                upper = upper * self.args.rho + (1 - self.args.rho)
        elif self.args.alloc_mode == 'root-inf':
            upper = 0
            for i in range(trained_num + 1):
                upper += 1/(torch.sqrt(torch.tensor(i + 1)))
            upper /= (math.pi **2 / 6)
            upper = min(1., upper)
        elif self.args.alloc_mode == 'uniform':
            if 'CIFAR100' in self.args.dataset:
                num = 100 // self.args.num_classes
                upper = (trained_num + 1)/num
            else:
                ValueError('TODO')
        elif self.args.alloc_mode == 'root':
            if 'CIFAR100' in self.args.dataset:
                num = 100 // self.args.num_classes
                total = 0
                for i in range(num):
                    total += 1/(torch.sqrt(torch.tensor(i + 1)))
                upper = 0
                for i in range(trained_num + 1):
                    upper += 1/(torch.sqrt(torch.tensor(i + 1)))
                upper /= total
            else:
                ValueError('TODO')
        elif self.args.alloc_mode == 'exp':
            if 'CIFAR100' in self.args.dataset:
                num = 100 // self.args.num_classes
                total = 0
                for i in range(num):
                    total += self.args.rho**i
                upper = 0
                for i in range(trained_num + 1):
                    upper += self.args.rho**i
                upper /= total
            else:
                ValueError('TODO')
        else:
            ValueError('TODO')
        res_sparsity_iter = {}
        for e in range(self.args.pruning_iter[0], self.args.pruning_iter[1]):
            res_sparsity_iter[e] = 1 - (e + 1 - self.args.pruning_iter[0])*(1-upper)/(self.args.pruning_iter[1]-self.args.pruning_iter[0])

        return res_sparsity_iter

    def prune(self, res_sparsity):
        # sparsity res
        for m in self.model.modules():
            if isinstance(m, Conv2d):
                '''
                thred = torch.topk((m.weight * (m.mask & m.free_conv_mask)).abs().view(-1), \
                    max(int(m.mask.numel() * res_sparsity - (~m.free_conv_mask).sum()),1))[0][-1]
                m.mask = (~m.free_conv_mask | (m.weight.abs() > thred)) & m.mask
                '''

                thred = torch.topk((m.weight * (m.mask & m.free_conv_mask)).abs().view(-1), \
                    max(math.ceil(m.free_conv_mask.sum() * res_sparsity),1))[0][-1]
                m.mask = (~m.free_conv_mask | (m.weight.abs() > thred)) & m.mask
        self.sparsity, _ = self.compute_sparsity()

    def warmup(self, epoch):
        if epoch < 1:
            return
        for m in self.model.modules():
            if isinstance(m, Conv2d):
                if m.free_conv_mask.sum() > 0:
                    m.mask = m.free_conv_mask
        for e in range(epoch):
            self.train(e, 'warmup')
        for m in self.model.modules():
            if isinstance(m, Conv2d):
                m.mask = torch.ones(m.mask.shape, dtype=torch.bool).cuda()

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
                for m in self.model.modules():
                    if isinstance(m, Conv2d):
                        m.weight.grad *= m.free_conv_mask
                
                self.optimizer.step()
        
                num = input.size(0)
                train_accuracy.update(classification_accuracy(output, target), num)
                train_loss.update(loss.cpu(), num)


                t.set_postfix({
                    'lr': '{:.2f}'.format(self.optimizer.param_groups[0]['lr']),
                    'loss': '{:.2f}'.format(train_loss.avg.item()),
                    'sparsity': '{:.2f}'.format(self.sparsity),
                    'acc': '{:.2f}'.format(100. * train_accuracy.avg.item()),
                    })
                t.update(1)
        if mode =='finetune':
            self.scheduler.step()
        summary = {
            'loss': train_loss.avg.item(),
            'acc': 100. * train_accuracy.avg.item()
            }
        self.validate()
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

                    t.set_postfix({ 
                        'loss': '{:.2f}'.format(loss.avg.item()),
                        'sparsity': '{:.2f}'.format(self.compute_sparsity()[0]),
                        'acc': '{:.2f}'.format(100. * accuracy.avg.item()),
                        })
                    t.update(1)

        summary = {
            'loss': loss.avg.item(),
            'acc': 100. * accuracy.avg.item(),
            }
        return summary
    
    def apply_task(self, task_id):
        self.data_loaders.update_task(task_id)
        self.model.eval()
        checkpoint = torch.load(self.filepath+'/{}.pth.tar'.format(task_id))
        running_means = checkpoint['running_means']
        running_vars = checkpoint['running_vars']
        num_batches_tracked = checkpoint['num_batches_tracked']
        masks = checkpoint['masks']
        header_weight = checkpoint['header_weight']
        header_bias = checkpoint['header_bias']

        for n, m in self.model.named_modules():
            if isinstance(m, BatchNorm2d):
                m.running_mean = running_means[n]
                m.running_var = running_vars[n]
                m.num_batches_tracked = num_batches_tracked[n]
            if isinstance(m, Conv2d):
                m.mask = masks[n]
        self.model.last.weight.data = header_weight
        if checkpoint['header_bias'] is not None:
            self.model.last.bias = header_bias
        
    def add_task(self, task_id):
        self.data_loaders.update_task(task_id)
        self.model.eval()
        self.init_weights()
        self.init_optimizer()
        for m in self.model.modules():
            if isinstance(m, Conv2d):
                m.mask = torch.ones(m.weight.shape, dtype=torch.bool).cuda()

    def save_checkpoint(self, task_id):
        running_means = {}
        running_vars = {}
        num_batches_tracked = {}
        masks = {}
        header_weight = self.model.last.weight
        header_bias = self.model.last.bias if self.model.last.bias is not None else None
        for n, m in self.model.named_modules():
            if isinstance(m, BatchNorm2d):
                running_means[n] = m.running_mean
                running_vars[n] = m.running_var
                num_batches_tracked[n] = m.num_batches_tracked
            if isinstance(m, Conv2d):
                masks[n] = m.mask
        checkpoint = {
            'running_means': running_means,
            'running_vars': running_vars,
            'num_batches_tracked': num_batches_tracked,
            'masks': masks,
            'header_weight': header_weight,
            'header_bias': header_bias,
        }
        torch.save(checkpoint, self.filepath + '/{}.pth.tar'.format(task_id))

    def compute_sparsity(self) -> float:
        sum = 0.
        count = 0.
        sum_all = 0.
        for m in self.model.modules():
            if isinstance(m, Conv2d):
                sum += m.mask.sum()
                count += m.mask.numel()
                sum_all += (~m.free_conv_mask).sum()
        return (sum / count).item(), (sum_all / count).item()