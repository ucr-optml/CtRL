import math
import models
from models_small.cell import Cell
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

        if 'MNIST' in self.args.dataset:
            self.input_size = [1,1,28,28]
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
        print('original FLOPs calculation: {}'.format(self.total_flops))
        if 'flop' in self.args.reg:
            self.update_base_flop()

    def update_base_flop(self):
        if 'flop' not in self.args.reg:
            return
        self.base_flop = 0.
        if self.args.reg == 'flop_0.5':
            for m in self.model.modules():
                if isinstance(m, Cell):
                    self.base_flop += math.sqrt((m.__flops_per__))
        else:
            raise ValueError('TODO : regularization {}.'.format(self.args.reg))
        self.base_flop /= self.layer_num

    def update_free_mask(self):
        for m in self.model.modules():
            if isinstance(m, Cell):
                m.free_mask &= ~m.mask
        self.model.free_mask &= ~self.model.mask

    def init_optimizer(self):
        optimizer_arg = {'params': self.model.parameters(),
                         'lr': self.args.lr}
        if 'Adam' in self.args.optimizer:
            optimizer_arg['betas'] = (0, 0.999)
        self.optimizer = torch.optim.__dict__[self.args.optimizer](**optimizer_arg)

    def forward(self, inputs, targets):
        out = self.model.forward(inputs)
        loss = self.criterion(out, targets)
        if self.reg == 'ori':
            sparsity_loss = 0.
        elif self.reg == 'l1':
            factors = torch.cat([m.value.view(-1) for m in self.model.modules() if isinstance(m, Cell)])
            num = torch.cat([m.get_output_mask() for m in self.model.modules() if isinstance(m, Cell)]).sum()
            sparsity_loss = 1. * torch.norm(factors, 1) / num  # factors.numel()
        elif self.reg == 'pol':
            sparsity_loss = 0.
            factors = torch.cat([m.value.view(-1) for m in self.model.modules() if isinstance(m, Cell)])
            num = torch.cat([m.get_output_mask() for m in self.model.modules() if isinstance(m, Cell)]).sum()
            sparse_weights_mean = factors.sum() / num  # factors.numel()
            for m in self.model.modules():
                if isinstance(m, Cell):
                    sparsity_term = 1.2 * torch.sum(torch.abs(m.value)) - torch.sum(
                        torch.abs(m.value - sparse_weights_mean))
                    sparsity_loss += 1.  * sparsity_term/ num  # factors.numel()
        elif self.reg == 'flop_0.5':
            factors = torch.cat([m.value.view(-1) * m.__flops_sqrt__ / self.base_flop \
                    for m in self.model.modules() if isinstance(m, Cell)])
            num = torch.cat([m.get_output_mask() for m in self.model.modules() if isinstance(m, Cell)]).sum()
            sparsity_loss = 1.  * torch.norm(factors, 1) / num  # factors.numel()
        else:
            raise ValueError('Sparsity regularizer {} is waiting to be done.'.format(self.config['reg']))

        loss += self.args.lam * sparsity_loss
        
        weights = torch.cat([(m.linear.weight * m.mask).view(-1) for m in self.model.modules() if isinstance(m, Cell)])
        # num = sum([m.mask.sum() for m in self.model.modules() if isinstance(m, Cell)])
        num = torch.cat([m.mask.view(-1) for m in self.model.modules() if isinstance(m, Cell)]).sum()
        l2_loss = 1. * torch.norm(weights, 2) / math.sqrt(num) # torch.sqrt(weights.numel())
        loss += self.args.lam * l2_loss

        self.optimizer.zero_grad()
        loss.backward()
        return loss.detach(), out

    def get_res_FLOP_iter(self):
        res_FLOP_iter = {}
        tgt = torch.sqrt(torch.tensor(self.args.res_FLOP))
        for e in range(self.args.pruning_iter[0], self.args.pruning_iter[1]):
            res_FLOP_iter[e] = (1-(e +1-self.args.pruning_iter[0])*(1-tgt)/(self.args.pruning_iter[1]-self.args.pruning_iter[0]))
            res_FLOP_iter[e] = res_FLOP_iter[e].item() ** 2
        return res_FLOP_iter

    def get_res_sparsity_iter(self):
        upper = self.args.rho
        res_sparsity_iter = {}
        for e in range(self.args.pruning_iter[0], self.args.pruning_iter[1]):
            res_sparsity_iter[e] = 1 - (e + 1 - self.args.pruning_iter[0])*(1-upper)/(self.args.pruning_iter[1]-self.args.pruning_iter[0])
        return res_sparsity_iter

    def prune(self, res_FLOP, res_sparsity):
        # flop res
        factors = torch.cat([m.value.view(-1).abs() for m in self.model.modules() if isinstance(m, Cell)])
        while (self.channel_ratio >0) and (self.flops_ratio > res_FLOP):
            self.channel_ratio -= 0.001
            thred = torch.topk(factors, max(int(factors.shape[0] * self.channel_ratio),1))[0][-1]
            for m in self.model.modules():
                if isinstance(m, Cell):
                    m.mask &= (m.value.abs() > thred).view(-1, 1)
            self.model.update_masks()
            self.flops_ratio = self.compute_flops() / self.total_flops
        self.update_base_flop()

        # sparsity res
        for m in self.model.modules():
            if isinstance(m, Cell):
                if (m.mask & m.free_mask).sum() <= math.ceil(res_sparsity*m.free_mask.sum()):
                    continue
                thred = torch.topk((m.linear.weight * (m.mask & m.free_mask)).abs().view(-1), \
                    max(math.ceil(m.free_mask.sum() * res_sparsity),1))[0][-1]
                m.mask = (~m.free_mask | (m.linear.weight.abs() > thred)) & m.mask

        if (self.model.mask & self.model.free_mask).sum() > math.ceil(res_sparsity*self.model.free_mask.sum()):
            thred = torch.topk((self.model.classifier.weight * (self.model.mask & self.model.free_mask)).abs().view(-1), \
                max(math.ceil(self.model.free_mask.sum() * res_sparsity),1))[0][-1]
            self.model.mask = (~self.model.free_mask | (self.model.classifier.weight.abs() > thred)) & self.model.mask
        self.sparsity, _ = self.compute_sparsity()

    def warmup(self, epoch):
        if epoch < 1:
            return
        # self.reg = 'ori'
        for m in self.model.modules():
            if isinstance(m, Cell):
                if m.free_mask.sum() > 0:
                    m.mask = m.free_mask
        if self.model.mask.sum() > 0:
            self.model.mask = self.model.free_mask
        for e in range(epoch):
            self.train(e, 'warmup')
        # self.reg = self.args.reg
        for m in self.model.modules():
            if isinstance(m, Cell):
                m.mask = torch.ones(m.mask.shape, dtype=torch.bool).cuda()
        self.model.mask = torch.ones(self.model.mask.shape, dtype=torch.bool).cuda()

    def train(self, epoch_idx, mode=None):
        if mode in ['warmup', 'finetune']:
            self.reg = 'ori'
        else:
            self.reg = self.args.reg
        if mode in ['finetune']:
            self.optimizer.param_groups[0]['lr'] = 0.0001
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
                loss, output = self.forward(input, target)
        
                # update grads
                for m in self.model.modules():
                    if isinstance(m, Cell):
                        m.linear.weight.grad *= (m.free_mask & m.mask)
                        m.value.grad *= m.get_output_mask()
                        # print(m.value)
                self.model.classifier.weight.grad *= (self.model.free_mask & self.model.mask)
                
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
        summary = {
            'loss': train_loss.avg.item(),
            'acc': 100. * train_accuracy.avg.item()
            }
        self.validate()
        return summary

    def validate(self, task_id=None):
        self.model.eval()
        if task_id is not None:
            self.set_task(task_id)
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
    
    def set_task(self, task_id):
        self.data_loaders.update_task(task_id)
        self.model.eval()
        checkpoint = torch.load(self.filepath+'/{}.pth.tar'.format(task_id))
        self.model.set_task(task_id, checkpoint)
        
    def add_task(self, task_id):
        self.data_loaders.update_task(task_id)
        self.channel_ratio = 1.
        self.init_optimizer()
        self.model.add_task(task_id)
        self.update_base_flop()
        
        self.sparsity, _ = self.compute_sparsity()
        self.flops_ratio = self.compute_flops() / self.total_flops
        print('Add task ID: {} with FLOPs: {}, sparsity: {}.'.format(task_id, self.flops_ratio, self.sparsity))

    def save_checkpoint(self, task_id):
        masks = {}
        for n, m in self.model.named_modules():
            if isinstance(m, Cell):
                masks[n] = m.mask
        masks['classifier'] = self.model.mask
        checkpoint = {
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
                sum_fixed += (~m.free_mask).sum()
                count += m.mask.numel()
        sum += self.model.mask.sum()
        sum_fixed += (~self.model.free_mask).sum()
        count += self.model.mask.numel()
        return (sum / count).item(), (sum_fixed / count).item()

    def compute_flops(self) -> float:
        global FLOPS
        FLOPS = 0.

        def cell_flops_counter_hook(m, input, output):
            global FLOPS
            input = input[0]
            
            in_neurons = m.get_input_mask().sum()
            out_neurons = m.get_output_mask().sum()
            flops = in_neurons * output.numel() * out_neurons // m.mask.shape[0]
            # cell flops
            m.__flops_per__ = flops // out_neurons if out_neurons != 0 else 0
            m.__flops_sqrt__ = math.sqrt(m.__flops_per__)
            FLOPS += flops

        def last_flops_counter_hook(m, input, output):
            global FLOPS
            input = input[0]

            in_neurons = (torch.sum(m.mask, (0))>0).sum()
            flops = in_neurons * output.numel()
            #
            FLOPS += flops

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