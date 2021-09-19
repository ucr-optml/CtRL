import torch
import torch.nn as nn
from torch.nn import Conv2d
import torch.nn.functional as F

class FC1024(nn.Module):
    def __init__(self, num_classes=10):
        super(FC1024, self).__init__()
        self.linear1 = Conv2d(28 * 28, 1024, kernel_size=1, padding=0, bias=False)
        self.linear2 = Conv2d(1024, 1024, kernel_size=1, padding=0, bias=False)
        self.last = nn.Linear(1024, num_classes, bias=False)
        self.last_input_mask = torch.ones(1024, dtype=torch.bool).cuda()
        self.last_mask = torch.ones(self.last.weight.size(), dtype=torch.bool).cuda()

    def forward(self, x):
        out = x.view(x.size(0), 28 * 28, 1, 1)
        w = self.linear1.weight * self.linear1.mask
        out = F.conv2d(out, w, self.linear1.bias, self.linear1.stride, self.linear1.padding, self.linear1.dilation, self.linear1.groups)
        out = F.relu(out)
        # out = F.relu(self.linear1(out))
        w = self.linear2.weight * self.linear2.mask
        out = F.conv2d(out, w, self.linear2.bias, self.linear2.stride, self.linear2.padding, self.linear2.dilation, self.linear2.groups)
        out = F.relu(out)
        # out = F.relu(self.linear2(out))
        out = torch.flatten(out, 1)
        out = self.logits(out)
        
        return out

    def logits(self, x):
        w = self.last.weight * self.last_mask
        x = F.linear(x, w, self.last.bias)
        return x

    def update_input_masks(self):
        self.linear2.input_mask = self.linear1.output_mask
        self.last_input_mask = self.linear2.output_mask
