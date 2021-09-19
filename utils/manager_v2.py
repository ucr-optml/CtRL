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
            ValueError('TODO for other datasets')
        self.total_flops = self.compute_flops()
        self.flops_ratio = 1.
        self.sparsity = 1.
        print('original FLOPs calculation: {}'.format(self.total_flops))
        if 'flop' in self.args.reg:
            self.update_base_flop()

    def update_base_flop(self):
        if 'flop' not in self.args.reg:
            return
        n = 0
        self.base_flop = 0.
        if self.args.reg == 'flop_0.5':
            for m in self.model.modules():
                if isinstance(m, Cell):
                    self.base_flop += torch.sqrt(m.__flops_per__)
                    n += 1
        else:
            ValueError('Regularization {} do not need to update flop!')
        self.base_flop /= n

    def update_free_conv_mask(self):
        for m in self.model.modules():
            if isinstance(m, Cell):
                m.free_conv_mask = (m.free_conv_mask & ~m.mask)

        ### 
        if self.args.share_header:
            self.model.free_last_mask = self.model.free_last_mask & ~self.model.last_mask

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
                if isinstance(m, Cell):
                    nn.init.ones_(m.bn.weight)
                    if m.bn.bias is not None:
                        nn.init.zeros_(m.bn.bias)
                    m.bn.running_mean.zero_()
                    m.bn.running_var.fill_(1)
                    m.bn.num_batches_tracked.zero_()

                    m.conv.weight.data = m.conv.weight.data*~m.free_conv_mask + nn.init.kaiming_normal_(m.conv.weight, mode='fan_out', nonlinearity='relu').cuda() * m.free_conv_mask
            ### nn.init.kaiming_uniform_(self.model.last.weight, a=math.sqrt(5))
            if self.model.last.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.model.last.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.model.last.bias, -bound, bound)
            
            ###
            if self.args.share_header:
                self.model.last.weight.data = self.model.last.weight.data * ~self.model.free_last_mask + \
                    nn.init.kaiming_uniform_(self.model.last.weight, a=math.sqrt(5)) * self.model.free_last_mask
            else:
                nn.init.kaiming_uniform_(self.model.last.weight, a=math.sqrt(5))

    def forward(self, inputs, targets):
        out = self.model.forward(inputs)
        loss = self.criterion(out, targets)
        if self.reg == 'ori':
            sparsity_loss = 0.
        elif self.reg == 'l1':
            factors = torch.cat([m.bn.weight.view(-1) for m in self.model.modules() if isinstance(m, Cell)])
            num = torch.cat([m.output_mask for m in self.model.modules() if isinstance(m, Cell)]).sum()
            sparsity_loss = 1. * torch.norm(factors, 1) / num # factors.numel()
        elif self.reg == 'pol':
            sparsity_loss = 0.
            # num = 0
            factors = torch.cat([m.bn.weight.view(-1) for m in self.model.modules() if isinstance(m, Cell)])
            num = torch.cat([m.output_mask for m in self.model.modules() if isinstance(m, Cell)]).sum()
            sparse_weights_mean = factors.sum() / num  # factors.numel()
            for m in self.model.modules():
                if isinstance(m, Cell):
                    sparsity_term = 1.2 * torch.sum(torch.abs(m.bn.weight)) - torch.sum(
                        torch.abs(m.bn.weight - sparse_weights_mean))
                    sparsity_loss += 1.  * sparsity_term/ num  # factors.numel()
        elif self.reg == 'flop_0.5':
            factors = torch.cat(
                [m.bn.weight.view(-1) * torch.sqrt(m.__flops_per__) / self.base_flop for m in self.model.modules() if
                 isinstance(m, Cell)])
            num = torch.cat([m.output_mask for m in self.model.modules() if isinstance(m, Cell)]).sum()
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

    def get_res_sparsity_iter(self, trained_num):
        if self.args.alloc_mode == 'exp-inf':
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

    def prune(self, res_FLOP, res_sparsity):
        # flop res
        factors = torch.cat([m.bn.weight.abs().view(-1) for m in self.model.modules() if isinstance(m, Cell)])
        while (self.channel_ratio >0) and (self.flops_ratio > res_FLOP):
            self.channel_ratio -= 0.001
            thred = torch.topk(factors, max(int(factors.shape[0] * self.channel_ratio),1))[0][-1]
            for m in self.model.modules():
                if isinstance(m, Cell):
                    m.output_mask = (m.bn.weight.abs() > thred)
            self.model.update_input_masks()
            self.flops_ratio = self.compute_flops() / self.total_flops
        for m in self.model.modules():
            if isinstance(m, Cell):
                m.bn.weight.data *= m.output_mask
                if m.bn.bias is not None:
                    m.bn.bias.data *= m.output_mask
                m.mask = m.mask * m.input_mask.view(1, -1, 1, 1) * m.output_mask.view(-1, 1, 1, 1)
        self.model.last_mask *= self.model.last_input_mask.view(1, -1)
        self.update_base_flop()

        # sparsity res
        for m in self.model.modules():
            if isinstance(m, Cell):
                # mask = m.free_conv_mask * m.input_mask.view(1, -1, 1, 1) * m.output_mask.view(-1, 1, 1, 1)
                # mask_fixed = ~m.free_conv_mask * m.input_mask.view(1, -1, 1, 1) * m.output_mask.view(-1, 1, 1, 1)
                # if (mask | ~m.free_conv_mask).sum() / mask.numel() <= res_sparsity:
                #     m.mask = mask | mask_fixed
                #     continue
                # thred = torch.topk((m.conv.weight * mask).abs().view(-1), max(int(mask.numel() * res_sparsity - (~m.free_conv_mask).sum()),1))[0][-1]
                # m.mask = mask_fixed | ((m.conv.weight.abs() > thred) & mask)
                if (m.mask | ~m.free_conv_mask).sum() / m.mask.numel() <= res_sparsity:
                    continue
                thred = torch.topk((m.conv.weight * (m.mask & m.free_conv_mask)).abs().view(-1), max(int(m.mask.numel() * res_sparsity - (~m.free_conv_mask).sum()),1))[0][-1]
                m.mask = (~m.free_conv_mask | (m.conv.weight.abs() > thred)) & m.mask
        ###
        if self.args.share_header:
            if (self.model.last_mask | ~self.model.free_last_mask).sum() / self.model.last_mask.numel() > res_sparsity:
                thred = torch.topk((self.model.last.weight * (self.model.last_mask & self.model.free_last_mask)).abs().view(-1), max(int(self.model.last_mask.numel() * res_sparsity - (~self.model.free_last_mask).sum()),1))[0][-1]
                self.model.last_mask = (~self.model.free_last_mask | (self.model.last.weight.abs() > thred)) & self.model.last_mask
        ###
        self.sparsity, _ = self.compute_sparsity()

    def warmup(self, epoch):
        if epoch < 1:
            return
        # self.reg = 'ori'
        for m in self.model.modules():
            if isinstance(m, Cell):
                if m.free_conv_mask.sum() > 0:
                    m.mask = m.free_conv_mask
        ###
        if self.args.share_header:
            if self.model.last_mask.sum() > 0:
                self.model.last_mask = self.model.free_last_mask
        ###
        for e in range(epoch):
            self.train(e, 'warmup')
        # self.reg = self.args.reg
        for m in self.model.modules():
            if isinstance(m, Cell):
                m.mask = torch.ones(m.mask.shape, dtype=torch.bool).cuda()
        ###
        if self.args.share_header:
            self.model.last_mask = torch.ones(self.model.last_mask.shape, dtype=torch.bool).cuda()
        ###

    def train(self, epoch_idx, mode=None):
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
            for batch_idx, (input, target) in enumerate(train_loader):
                input, target = input.cuda(), target.cuda()
                loss, output = self.forward(input, target)
        
                # update grads
                for m in self.model.modules():
                    if isinstance(m, Cell):
                        m.conv.weight.grad *= m.free_conv_mask
                        m.bn.weight.grad *= m.output_mask
                        if m.bn.bias is not None:
                            m.bn.bias.grad *= m.output_mask
                self.model.last.weight.grad *= self.model.last_mask
                ###
                if self.args.share_header:
                    self.model.last.weight.grad *= self.model.free_last_mask
                    if self.model.last.bias is not None:
                        self.model.last.bias.grad *= 0
                ###
                
                self.optimizer.step()
        
                num = input.size(0)
                train_accuracy.update(classification_accuracy(output, target), num)
                train_loss.update(loss.cpu(), num)

                t.set_postfix({
                    'lr': '{:.2f}'.format(self.optimizer.param_groups[0]['lr']),
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
                        'FLOPs': '{:.2f}'.format((self.compute_flops()/self.total_flops)),
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
        channel_weights = checkpoint['channel_weights']
        channel_biases = checkpoint['channel_biases']
        running_means = checkpoint['running_means']
        running_vars = checkpoint['running_vars']
        num_batches_tracked = checkpoint['num_batches_tracked']
        channel_out_masks = checkpoint['channel_out_masks']
        masks = checkpoint['masks']
        header_weight = checkpoint['header_weight']
        header_bias = checkpoint['header_bias']
        header_mask = checkpoint['header_mask']

        for n, m in self.model.named_modules():
            if isinstance(m, Cell):
                m.bn.weight.data = channel_weights[n]
                if m.bn.bias is not None:
                    m.bn.bias.data = channel_biases[n]
                m.bn.running_mean = running_means[n]
                m.bn.running_var = running_vars[n]
                m.bn.num_batches_tracked = num_batches_tracked[n]
                m.output_mask = channel_out_masks[n]
                m.mask = masks[n]
        self.model.update_input_masks()
        ### self.model.last.weight.data = header_weight
        ### 1
        if checkpoint['header_weight'] is not None:
            self.model.last.weight.data = header_weight
        ### 2
        if checkpoint['header_bias'] is not None:
            self.model.last.bias = header_bias
        self.model.last_mask = header_mask
        
    def add_task(self, task_id):
        self.data_loaders.update_task(task_id)
        self.channel_ratio = 1.
        self.model.eval()
        self.init_weights()
        self.init_optimizer()
        for m in self.model.modules():
            if isinstance(m, Cell):
                m.output_mask = torch.ones(m.output_mask.shape, dtype=torch.bool).cuda()
                m.mask = torch.ones(m.mask.shape, dtype=torch.bool).cuda()
        self.model.update_input_masks()
        self.update_base_flop()

        self.model.last_mask = torch.ones(self.model.last_mask.shape, dtype=torch.bool).cuda()
        self.model.last_input_mask = torch.ones(self.model.last_input_mask.shape, dtype=torch.bool).cuda()
        self.sparsity, _ = self.compute_sparsity()
        self.flops_ratio = self.compute_flops() / self.total_flops
        print('Add task ID: {} with FLOPs: {}, sparsity: {}.'.format(task_id, self.flops_ratio, self.sparsity))

    def save_checkpoint(self, task_id):
        channel_weights = {}
        channel_biases = {}
        running_means = {}
        running_vars = {}
        num_batches_tracked = {}
        channel_out_masks = {}
        masks = {}
        header_weight = self.model.last.weight * self.model.last_mask
        header_bias = self.model.last.bias if self.model.last.bias is not None else None
        header_mask = self.model.last_mask
        for n, m in self.model.named_modules():
            if isinstance(m, Cell):
                channel_weights[n] = m.bn.weight * m.output_mask
                if m.bn.bias is not None:
                    channel_biases[n] = m.bn.bias * m.output_mask
                running_means[n] = m.bn.running_mean * m.output_mask
                running_vars[n] = m.bn.running_var * m.output_mask
                num_batches_tracked[n] = m.bn.num_batches_tracked
                channel_out_masks[n] = m.output_mask
                masks[n] = m.mask
        ### 1
        if self.args.share_header:
            header_weight = None
            header_bias = None
        ### 2
        checkpoint = {
            'channel_weights': channel_weights,
            'channel_biases': channel_biases,
            'running_means': running_means,
            'running_vars': running_vars,
            'num_batches_tracked': num_batches_tracked,
            'channel_out_masks': channel_out_masks,
            'masks': masks,
            'header_weight': header_weight,
            'header_bias': header_bias,
            'header_mask': header_mask,
        }
        torch.save(checkpoint, self.filepath + '/{}.pth.tar'.format(task_id))
        
    def compute_sparsity(self) -> float:
        sum = 0.
        count = 0.
        sum_all = 0.
        for m in self.model.modules():
            if isinstance(m, Cell):
                sum += m.mask.sum()
                count += m.mask.numel()
                sum_all += (~m.free_conv_mask).sum()
        ### 1
        if self.args.share_header:
            sum += self.model.last_mask.sum()
            count += self.model.last_mask.numel()
            sum_all += (~self.model.free_last_mask).sum()
        ### 2
        return (sum / count).item(), (sum_all / count).item()

    def compute_flops(self) -> float:
        global FLOPS
        FLOPS = 0.

        def cell_flops_counter_hook(m, input, output):
            global FLOPS
            input = input[0]
            # conv flops
            batch_size = input.shape[0]
            output_dims = output.shape[2:]

            kernel_dims = m.conv.kernel_size
            in_channels = m.conv.in_channels
            out_channels = m.conv.out_channels
            groups = m.conv.groups

            filters_per_channel = out_channels // groups
            conv_per_position_flops = int(torch.prod(torch.tensor(kernel_dims))) * \
                                    in_channels * filters_per_channel

            active_elements_count = batch_size * int(torch.prod(torch.tensor(output_dims)))

            overall_conv_flops = conv_per_position_flops * active_elements_count

            bias_flops = 0

            if m.conv.bias is not None:
                bias_flops = out_channels * active_elements_count

            overall_flops = overall_conv_flops + bias_flops

            # bn flops
            batch_flops = torch.prod(torch.tensor(output[0].shape)).to(torch.float)
            if m.bn.affine:
                batch_flops *= 2

            # cell flops
            in_ratio = m.input_mask.sum().to(torch.float) / m.input_mask.numel()
            out_ratio = m.output_mask.sum().to(torch.float) / m.output_mask.numel()
            in_ratio, out_ratio = in_ratio.cpu(), out_ratio.cpu()
            overall_flops = overall_conv_flops * in_ratio * out_ratio + bias_flops * out_ratio
            batch_flops *= out_ratio
            # if hasattr('__flops_per__'):
            #     m.__flops_per_full__ = m.__flops_per__
            #     m.__flops_per__ = (overall_flops + batch_flops) / (m.output_mask.sum() + 1e-5)
            # else:
            #     m.__setattr__('__flops_per_full__', (overall_flops + batch_flops) / m.bn.weight.numel())
            #     m.__setattr__('__flops_per__', m.__flops_per_full__)
            m.__flops_per__ = (overall_flops + batch_flops) / (m.output_mask.sum() + 1e-5)
            FLOPS += (overall_flops + batch_flops)

        def last_flops_counter_hook(m, input, output):
            global FLOPS
            input = input[0]
            # linear flops
            output_last_dim = output.shape[-1]
            bias_flops = output_last_dim  if m.last.bias is not None else 0
            overall_flops = int(torch.prod(torch.tensor(input.shape)) * output_last_dim + bias_flops)

            #
            in_ratio = m.last_input_mask.sum().to(torch.float) / m.last_input_mask.numel()
            in_ratio = in_ratio.cpu()
            overall_flops *= in_ratio
            FLOPS += overall_flops

        def add_hooks(net, hook_handles):
            hook_handles.append(net.register_forward_hook(last_flops_counter_hook))
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