import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import Cell

class FC1024(nn.Module):
    def __init__(self, num_classes=10):
        super(FC1024, self).__init__()
        self.linear1 = Cell(28 * 28, 1024)
        self.linear2 = Cell(1024, 1024)
        self.classifier = nn.Linear(1024, num_classes, bias=False)
        self.task_hist = {}
        self.mask = torch.ones(self.classifier.weight.size(), dtype=torch.bool).cuda()
        self.free_mask = torch.ones(self.classifier.weight.size(), dtype=torch.bool).cuda()

    def _init_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, Cell):
                    nn.init.ones_(m.value)
                    m.value.requires_grad_(True)
                    m.linear.weight.data = m.linear.weight.data * ~m.free_mask + \
                        nn.init.kaiming_uniform_(m.linear.weight, a=math.sqrt(5)) * m.free_mask
            self.classifier.weight.data = self.classifier.weight.data * ~self.free_mask + \
                nn.init.kaiming_uniform_(self.classifier.weight, a=math.sqrt(5)) * self.free_mask          

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.linear1(out))
        out = F.relu(self.linear2(out))
        out = self.logits(out)        
        return out

    def logits(self, x):
        w = self.classifier.weight * self.mask
        x = F.linear(x, w, self.classifier.bias)
        return x

    def update_masks(self):
        self.linear2.mask &= self.linear1.get_output_mask().view(1, -1)
        self.mask &= self.linear2.get_output_mask().view(1, -1)

        for m in self.modules():
            if isinstance(m, Cell):
                m.value.data *= m.get_output_mask()
                
    def add_task(self, task_id):
        self.task_hist[str(task_id)] = {}
        self.task_hist[str(task_id)]['order'] = len(self.task_hist.keys())
        for m in self.modules():
            if isinstance(m, Cell):
                m.mask = torch.ones(m.linear.weight.shape, dtype=torch.bool).cuda()
        self.mask = torch.ones(self.classifier.weight.shape, dtype=torch.bool).cuda()
        self._init_weights()

    def set_task(self, task_id, cp):
        if not str(task_id) in self.task_hist.keys():
            raise ValueError('task id {} is not trained!'.format(task_id))
        masks = cp['masks']
        for n, m in self.named_modules():
            if isinstance(m, Cell):
                m.mask = masks[n]
        self.mask = masks['classifier']
        
    def load_state_dict(self, filename):
        checkpoint = torch.load(filename)
        task_hist = checkpoint['task_hist']
        state_dict = checkpoint['state_dict']
        free_masks = checkpoint['free_masks']
        super().load_state_dict(state_dict)
        self.task_hist = task_hist
        for n, m in self.named_modules():
            if isinstance(m, Cell):
                m.free_mask = free_masks[n]
    
    def save_state_dict(self, filename):
        free_masks = {}
        for n, m in self.named_modules():
            if isinstance(m, Cell):
                free_masks[n] = m.free_mask
        checkpoint = {}
        checkpoint['task_hist'] = self.task_hist
        checkpoint['state_dict'] = self.state_dict()
        checkpoint['free_masks'] = free_masks
        torch.save(checkpoint, filename)
    
    def task_exists(self, task_id):
        return str(task_id) in self.task_hist.keys()

