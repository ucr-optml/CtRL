from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd



class Cell(nn.Module):

    def __init__(self, inplanes, planes,
                 stride: int = 1,
                 groups: int = 1,
                 dilation: int = 1,
                 kernel_size: int = 3,
                 padding: int = 1, 
                 bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.conv = nn.Conv2d(inplanes, planes, kernel_size, stride,
                              groups=groups, dilation=dilation, padding=padding, bias=bias)

        self.value = nn.Parameter(torch.Tensor(planes))

        self.input_mask = torch.ones(inplanes, dtype=torch.bool).cuda()
        self.output_mask = torch.ones(planes, dtype=torch.bool).cuda()
        self.mask = torch.ones(self.conv.weight.size(), dtype=torch.bool).cuda()

        self.free_conv_mask = torch.ones(self.conv.weight.shape, dtype=torch.bool).cuda()




    def forward(self, x):
        w = self.conv.weight * self.mask
        x = F.conv2d(
            x, w, self.conv.bias, self.stride, self.padding, self.dilation, self.groups
        )
        x = x * (self.output_mask * self.value).view(1,-1, 1, 1)
        return x
