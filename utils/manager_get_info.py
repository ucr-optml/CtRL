from models.cell import Cell
from re import S
import math
from torch._C import TracingState
from torchvision import transforms
from models.cell import Cell
from torch.nn import Conv2d, Linear, BatchNorm2d
import torch
import torch.nn as nn
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

        self.ratio_fixed = 1.

        self.masks = {}
        for n, m in self.model.named_modules():
            if isinstance(m, Cell):
                self.masks[n] = torch.ones(m.conv.weight.shape, dtype=torch.bool).cuda()

        

    def get_info(self, task_id):
        self.apply_task(task_id)
        count = 0.
        num = 0.
        num_fixed = 0.
        num_free = 0.
        for n, m in self.model.named_modules():
            if isinstance(m, Cell):
                num_fixed += ((~self.masks[n] & m.input_mask.view([1, -1, 1, 1])) \
                    & (~self.masks[n] & m.output_mask.view([-1, 1, 1, 1]))).sum()
                self.masks[n] = (self.masks[n] & ~m.input_mask.view([1, -1, 1, 1])) \
                    | (self.masks[n] & ~m.output_mask.view([-1, 1, 1, 1]))
                # count_free += self.masks[n].numel()
                num_free += self.masks[n].sum()
                count += m.conv.weight.numel()
                num += m.conv.weight.numel() * m.input_mask.sum() * m.output_mask.sum() / m.input_mask.numel() / m.output_mask.numel()
        # count += self.model.last.weight.numel()
        num += self.model.last.weight.numel() * self.model.last_mask.sum() / self.model.last_mask.numel()
        return num/(count + self.model.last.weight.numel()), num_free / count, num_fixed / count

        
    def apply_task(self, task_id):
        self.model.eval()
        checkpoint = torch.load(self.filepath+'/{}.pth.tar'.format(task_id))
        channel_weights = checkpoint['channel_weights']
        channel_biases = checkpoint['channel_biases']
        running_means = checkpoint['running_means']
        running_vars = checkpoint['running_vars']
        num_batches_tracked = checkpoint['num_batches_tracked']
        channel_in_masks = checkpoint['channel_in_masks']
        channel_out_masks = checkpoint['channel_out_masks']
        header_weight = checkpoint['header_weight']
        header_bias = checkpoint['header_bias']
        header_mask = checkpoint['header_mask']

        with torch.no_grad():
            for n, m in self.model.named_modules():
                if isinstance(m, Cell):
                    m.bn.weight = torch.nn.Parameter(channel_weights[n]).cuda()
                    if m.bn.bias is not None:
                        m.bn.bias = torch.nn.Parameter(channel_biases[n]).cuda()
                    m.bn.running_mean = running_means[n].cuda()
                    m.bn.running_var = running_vars[n].cuda()
                    m.bn.num_batches_tracked = num_batches_tracked[n].cuda()
                    m.input_mask = channel_in_masks[n].cuda()
                    m.output_mask = channel_out_masks[n].cuda()
        self.model.last.weight = nn.Parameter(header_weight).cuda()
        if checkpoint['header_bias'] is not None:
            self.model.last.bias = nn.Parameter(header_bias).cuda()
        self.model.last_mask = header_mask.cuda()         

                   


def compute_sparsity(masks) -> float:

    sum = 0.
    count = 0.
    for m in masks.values():
        sum += m.sum()
        count += m.numel()
    return sum / count