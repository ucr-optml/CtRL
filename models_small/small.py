import torch
import torch.nn as nn
import torch.nn.functional as F
from . import Cell

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.linear1 = Cell(28 * 28, 300, kernel_size=1, padding=0)
        self.linear2 = Cell(300, 100, kernel_size=1, padding=0)
        self.last = nn.Linear(100, num_classes, bias=False)
        self.last_input_mask = torch.ones(100, dtype=torch.bool).cuda()
        self.last_mask = torch.ones(self.last.weight.size(), dtype=torch.bool).cuda()

    def forward(self, x):
        out = x.view(x.size(0), 28 * 28, 1, 1)
        out = F.relu(self.linear1(out))
        out = F.relu(self.linear2(out))
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

class FC1024(nn.Module):
    def __init__(self, num_classes=10):
        super(FC1024, self).__init__()
        self.linear1 = Cell(28 * 28, 1024, kernel_size=1, padding=0)
        self.linear2 = Cell(1024, 1024, kernel_size=1, padding=0)
        self.last = nn.Linear(1024, num_classes, bias=False)
        self.last_input_mask = torch.ones(1024, dtype=torch.bool).cuda()
        self.last_mask = torch.ones(self.last.weight.size(), dtype=torch.bool).cuda()

    def forward(self, x):
        out = x.view(x.size(0), 28 * 28, 1, 1)
        out = F.relu(self.linear1(out))
        out = F.relu(self.linear2(out))
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

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = Cell(1, 6, kernel_size=5, padding=2)
        self.conv2 = Cell(6, 16, kernel_size=5, padding=0)
        self.conv3 = Cell(16, 120, kernel_size=5, padding=0)
        self.linear1 = Cell(120, 84, kernel_size=1, padding=0)
        self.last = nn.Linear(84, num_classes, bias=False)
        self.last_input_mask = torch.ones(1024, dtype=torch.bool).cuda()
        self.last_mask = torch.ones(self.last.weight.size(), dtype=torch.bool).cuda()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = x
        out = self.maxpool(F.relu(self.conv1(out)))
        out = self.maxpool(F.relu(self.conv2(out)))
        out = F.relu(self.conv3(out))
        out = F.relu(self.linear1(out))
        out = torch.flatten(out, 1)
        out = self.logits(out)
        
        return out

    def logits(self, x):
        w = self.last.weight * self.last_mask
        x = F.linear(x, w, self.last.bias)
        return x

    def update_input_masks(self):
        self.conv2.input_mask = self.conv1.output_mask
        self.conv3.input_mask = self.conv2.output_mask
        self.linear1.input_mask = self.conv3.output_mask
        self.last_input_mask = self.linear1.output_mask