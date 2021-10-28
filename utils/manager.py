import math
from models.cell import Cell
import torch
import torch.nn as nn
from tqdm import tqdm
from . import Metric, classification_accuracy



class Manager(object):

    
    def __init__(self, args, model, data_loaders, filepath) -> None:
        super().__init__()
        self.args = args
        self.model = model
        self.data_loaders = data_loaders
        self.filepath = filepath
        self.reg = self.args.reg

        self.channel_ratio = 1.
        self.init_optimizer()
        self.criterion = nn.CrossEntropyLoss()

        if 'CIFAR' in self.args.dataset:
            self.input_size = [1,3,32,32]
        elif 'ImageNet' in self.args.dataset:
            self.input_size = [1,3,224,224]
        else:
            raise ValueError('TODO for other datasets')
        self.total_flops = self.compute_flops()
        self.flops_ratio = 1.
        self.sparsity = 1.
        self.layer_num = 0
        for m in self.model.modules():
                if isinstance(m, Cell):
                    self.layer_num += 1
        print('Layer number: {}'.format(self.layer_num))
        print('Original FLOPs calculation: {}'.format(self.total_flops))
        if 'flop' in self.args.reg:
            self.update_base_flop()

    def update_base_flop(self):
        if 'flop' not in self.args.reg:
            return
        self.base_flop = 0.
        if self.args.reg == 'flop_0.5':
            for m in self.model.modules():
                if isinstance(m, Cell):
                    self.base_flop += m.__flops_sqrt__
        else:
            raise ValueError('TODO : regularization {}.'.format(self.args.reg))
        self.base_flop /= self.layer_num

    def update_free_conv_mask(self):
        for m in self.model.modules():
            if isinstance(m, Cell):
                m.free_conv_mask = (m.free_conv_mask & ~m.mask)

    def init_optimizer(self):
        optimizer_arg = {'params':self.model.parameters(),
                         'lr':self.args.lr}
        if 'Adam' in self.args.optimizer:
            optimizer_arg['betas'] = (0, 0.999)
        self.optimizer = torch.optim.__dict__[self.args.optimizer](**optimizer_arg)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.epochs - self.args.pruning_iter[-1])
        
        def warm_up_with_multistep_lr(epoch): 
            return (epoch+1) / self.args.warmup if epoch < self.args.warmup \
            else 1 
        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warm_up_with_multistep_lr)

    def forward(self, inputs, targets, task_id):
        out = self.model.forward(inputs, task_id)
        loss = self.criterion(out, targets)
        if self.reg == 'ori':
            sparsity_loss = 0.
        elif self.reg == 'l1':
            factors = torch.cat([m.bn.weight.view(-1) for m in self.model.modules() if isinstance(m, Cell)])
            num = torch.cat([m.get_output_mask() for m in self.model.modules() if isinstance(m, Cell)]).sum()
            sparsity_loss = 1. * torch.norm(factors, 1) / num # factors.numel()
        elif self.reg == 'pol':
            sparsity_loss = 0.
            # num = 0
            factors = torch.cat([m.bn.weight.view(-1) for m in self.model.modules() if isinstance(m, Cell)])
            num = torch.cat([m.get_output_mask() for m in self.model.modules() if isinstance(m, Cell)]).sum()
            sparse_weights_mean = factors.sum() / num  # factors.numel()
            for m in self.model.modules():
                if isinstance(m, Cell):
                    sparsity_term = 1.2 * torch.sum(torch.abs(m.bn.weight)) - torch.sum(
                        torch.abs(m.bn.weight - sparse_weights_mean))
                    sparsity_loss += 1.  * sparsity_term/ num  # factors.numel()
        elif self.reg == 'flop_0.5':
            factors = torch.cat(
                [m.bn.weight.view(-1) * m.__flops_sqrt__ / self.base_flop for m in self.model.modules() if
                 isinstance(m, Cell)])
            num = torch.cat([m.get_output_mask() for m in self.model.modules() if isinstance(m, Cell)]).sum()
            sparsity_loss = 1.  * torch.norm(factors, 1)/ num  # factors.numel()
        else:
            ValueError('reg {} does not exist!'.format(self.config['reg']))

        loss += self.args.lam * sparsity_loss
        
        weights = torch.cat([(m.conv.weight * m.mask).view(-1) for m in self.model.modules() if isinstance(m, Cell)])
        num = torch.cat([m.mask.view(-1) for m in self.model.modules() if isinstance(m, Cell)]).sum()
        l2_loss = 1./torch.sqrt(num) * torch.norm(weights, 2)
        loss += self.args.lam * l2_loss

        self.optimizer.zero_grad()
        loss.backward()
        return loss.detach(), out
    
    def get_res_FLOP_iter(self):
        res_FLOP_iter = {}
        tgt = torch.sqrt(torch.tensor(self.args.res_FLOP))
        for e in range(self.args.pruning_iter[0], self.args.pruning_iter[1]):
            res_FLOP_iter[e] = 1 - (e + 1 - self.args.pruning_iter[0])*(1-tgt)/(self.args.pruning_iter[1]-self.args.pruning_iter[0])
            res_FLOP_iter[e] = res_FLOP_iter[e].item() ** 2
        return res_FLOP_iter

    def get_res_sparsity_iter(self):
        upper = self.args.rho
        res_sparsity_iter = {}
        for e in range(self.args.pruning_iter[0], self.args.pruning_iter[1]):
            res_sparsity_iter[e] = 1 - (e + 1 - self.args.pruning_iter[0])*(1-upper)/(self.args.pruning_iter[1]-self.args.pruning_iter[0])
        return res_sparsity_iter

    def prune(self, task_id, res_FLOP, res_sparsity):
        # flop res
        factors = torch.cat([m.bn.weight.abs().view(-1) for m in self.model.modules() if isinstance(m, Cell)])
        while (self.channel_ratio >0) and (self.flops_ratio > res_FLOP):
            self.channel_ratio -= 0.001
            thred = torch.topk(factors, max(int(factors.shape[0] * self.channel_ratio),1))[0][-1]
            for m in self.model.modules():
                if isinstance(m, Cell):
                    m.mask &= (m.bn.weight.abs() > thred).view(-1, 1, 1, 1)
            self.model.update_masks()
            self.flops_ratio = self.compute_flops() / self.total_flops
        
        # for m in self.model.modules():
        #     if isinstance(m, Cell):
        #         m.bn.weight.data *= m.get_output_mask()
        #         if m.bn.bias is not None:
        #             m.bn.bias.data *= m.get_output_mask()
        # self.model.classifiers[str(task_id)].weight.data *= self.mask

        self.update_base_flop()

        # sparsity res
        for m in self.model.modules():
            if isinstance(m, Cell):
                if (m.mask & m.free_conv_mask).sum() <= math.ceil(res_sparsity*m.free_conv_mask.sum()):
                    continue
                thred = torch.topk((m.conv.weight * (m.mask & m.free_conv_mask)).abs().view(-1), \
                    max(math.ceil(m.free_conv_mask.sum() * res_sparsity),1))[0][-1]
                m.mask = (~m.free_conv_mask | (m.conv.weight.abs() > thred)) & m.mask
        self.sparsity, _ = self.compute_sparsity()

    def warmup(self, epoch, task_id):
        if epoch < 1:
            return
        for m in self.model.modules():
            if isinstance(m, Cell):
                if m.free_conv_mask.sum() > 0:
                    m.mask = m.free_conv_mask
        for e in range(epoch):
            self.train(e, task_id, mode='warmup')
            self.warmup_scheduler.step()
        for m in self.model.modules():
            if isinstance(m, Cell):
                m.mask = torch.ones(m.mask.shape, dtype=torch.bool).cuda()

    def train(self, epoch_idx, task_id, mode=None):
        if mode in ['warmup', 'finetune']:
            self.reg = 'ori'
        else:
            self.reg = self.args.reg
        self.model.train()
        train_loader = self.data_loaders.train_loader
        train_loss = Metric('train_loss')
        train_accuracy = Metric('train_accuracy')

        with tqdm(total=len(train_loader),
                  desc='Train Ep. #{}: '.format(epoch_idx + 1),
                  disable=False,
                  ascii=True) as t:
            for input, target in train_loader:
                input, target = input.cuda(), target.cuda()
                loss, output = self.forward(input, target, task_id)
        
                # update grads
                for m in self.model.modules():
                    if isinstance(m, Cell):
                        m.conv.weight.grad *= (m.free_conv_mask & m.mask)
                        m.bn.weight.grad *= m.get_output_mask()
                        if m.bn.bias is not None:
                            m.bn.bias.grad *= m.get_output_mask()
                self.model.classifiers[str(task_id)].weight.grad *= self.model.mask
                
                self.optimizer.step()
        
                num = input.size(0)
                train_accuracy.update(classification_accuracy(output, target), num)
                train_loss.update(loss.cpu(), num)

                t.set_postfix({
                    'lr': '{:.3f}'.format(self.optimizer.param_groups[0]['lr']),
                    'loss': '{:.2f}'.format(train_loss.avg.item()),
                    'sparsity': '{:.2f}'.format(self.sparsity),
                    'FLOPs': '{:.2f}'.format(self.flops_ratio),
                    'acc': '{:.2f}'.format(100. * train_accuracy.avg.item()),
                    })
                t.update(1)
        if mode =='finetune':
            self.scheduler.step()
        summary = {
            'loss': train_loss.avg.item(),
            'acc': 100. * train_accuracy.avg.item()
            }
        self.validate(task_id)
        return summary

    def validate(self, task_id):
        self.model.eval()
        val_loader = self.data_loaders.val_loader
        loss = Metric('loss')
        accuracy = Metric('accuracy')

        with tqdm(total=len(val_loader),
                  desc='Val: ',
                  ascii=True) as t:
            for data, target in val_loader:
                data, target = data.cuda(), target.cuda()

                output = self.model(data, task_id)
                num = data.size(0)
                loss.update(self.criterion(output, target).cpu(), num)
                accuracy.update(classification_accuracy(output, target), num)

                t.set_postfix({ 
                    'loss': '{:.3f}'.format(loss.avg.item()),
                    'sparsity': '{:.2f}'.format(self.compute_sparsity()[0]),
                    'FLOPs': '{:.2f}'.format((self.compute_flops()/self.total_flops)),
                    'acc': '{:.2f}'.format(100. * accuracy.avg.item()),
                    })
                t.update(1)

        summary = {
            'loss': loss.avg.item(),
            'acc': 100. * accuracy.avg.item(),
            }
        return summary
    
    def set_task(self, task_id):
        self.data_loaders.update_task(task_id)
        self.model.eval()
        checkpoint = torch.load(self.filepath+'/{}.pth.tar'.format(task_id))
        self.model.set_task(task_id, checkpoint)
        
    def add_task(self, task_id, num_classes):
        self.data_loaders.update_task(task_id)
        self.channel_ratio = 1.
        self.init_optimizer()
        self.model.add_task(task_id, num_classes)
        self.update_base_flop()

        self.sparsity, _ = self.compute_sparsity()
        self.flops_ratio = self.compute_flops() / self.total_flops
        print('Add task ID: {} with FLOPs: {}, sparsity: {}.'.format(task_id, self.flops_ratio, self.sparsity))

    def save_checkpoint(self, task_id):
        channel_weights = {}
        channel_biases = {}
        running_means = {}
        running_vars = {}
        num_batches_tracked = {}
        masks = {}
        for n, m in self.model.named_modules():
            if isinstance(m, Cell):
                channel_weights[n] = m.bn.weight * m.get_output_mask()
                if m.bn.bias is not None:
                    channel_biases[n] = m.bn.bias * m.get_output_mask()
                running_means[n] = m.bn.running_mean * m.get_output_mask()
                running_vars[n] = m.bn.running_var * m.get_output_mask()
                num_batches_tracked[n] = m.bn.num_batches_tracked
                masks[n] = m.mask
        checkpoint = {
            'channel_weights': channel_weights,
            'channel_biases': channel_biases,
            'running_means': running_means,
            'running_vars': running_vars,
            'num_batches_tracked': num_batches_tracked,
            'masks': masks,
        }
        torch.save(checkpoint, self.filepath + '/{}.pth.tar'.format(task_id))
        
    def compute_sparsity(self) -> float:
        sum = 0.
        sum_fixed = 0.
        count = 0
        for m in self.model.modules():
            if isinstance(m, Cell):
                sum += m.mask.sum()
                sum_fixed += (~m.free_conv_mask).sum()
                count += m.mask.numel()
        return (sum / count).item(), (sum_fixed / count).item()

    def compute_flops(self) -> float:
        global FLOPS
        FLOPS = 0.

        def cell_flops_counter_hook(m, input, output):
            global FLOPS
            input = input[0]

            # conv flops
            kernel_dims = m.conv.kernel_size
            in_channels = m.get_input_mask().sum()
            out_channels = m.get_output_mask().sum()
            groups = m.conv.groups
            conv_per_position_flops = int(torch.prod(torch.tensor(kernel_dims))) * in_channels // groups
            conv_flops = conv_per_position_flops * output.numel() * out_channels // m.conv.out_channels

            # bn flops
            bn_flops = output.numel() * out_channels // m.conv.out_channels
            if m.bn.track_running_stats:
                bn_flops *= 2

            # cell flops
            overall_flops = conv_flops + bn_flops

            m.__flops_per__ = overall_flops // out_channels if out_channels != 0 else 0
            m.__flops_sqrt__ = math.sqrt(m.__flops_per__)
            FLOPS += overall_flops

        def add_hooks(net, hook_handles):
            for net in net.modules():
                if isinstance(net, Cell):
                    hook_handles.append(net.register_forward_hook(cell_flops_counter_hook))
            return

        handles = []
        add_hooks(self.model, handles)
        demo_input = torch.rand(self.input_size).cuda()
        self.model(demo_input)
        # clear handles
        for h in handles:
            h.remove()

        return FLOPS.item()