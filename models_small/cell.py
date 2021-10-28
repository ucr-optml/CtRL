from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd



class Cell(nn.Module):

    def __init__(self, inplanes, planes,
                 bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(inplanes, planes, bias=bias)
        self.value = nn.Parameter(torch.Tensor(planes))

        self.mask = torch.ones(self.linear.weight.size(), dtype=torch.bool).cuda()
        self.free_mask = torch.ones(self.linear.weight.shape, dtype=torch.bool).cuda()

    def forward(self, x):
        w = self.linear.weight * self.mask
        x = F.linear(
            x, w, self.linear.bias
        )
        x = x * (self.get_output_mask() * self.value).view(1,-1)
        return x

    def get_input_mask(self):
        return torch.sum(self.mask, (0))>0
    
    def get_output_mask(self):
        return torch.sum(self.mask, (1))>0