import math
import torch
import torch.nn as nn
from torch.nn import Conv2d, Linear, BatchNorm2d
from tqdm import tqdm
from . import Metric

import numpy as np

class Manager(object):

    
    def __init__(self, args, model, train_loader, val_loader, filepath) -> None:
        super().__init__()
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.filepath = filepath

        self.init_optimizer()
        self.criterion = nn.CrossEntropyLoss()

        np.random.seed(args.seed_data)
        perm = np.random.permutation(100)
        self.labels = [perm[args.num_classes * i : args.num_classes * (i + 1)] for i in range(100//args.num_classes)]
        self.labels = torch.tensor(self.labels)
        print(self.labels)

    def init_optimizer(self):
        optimizer_arg = {'params':self.model.parameters(),
                         'lr':self.args.lr}
        if 'Adam' in self.args.optimizer:
            optimizer_arg['betas'] = (0, 0.999)
        self.optimizer = torch.optim.__dict__[self.args.optimizer](**optimizer_arg)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.epochs - 150)

    def init_weights(self):
        with torch.no_grad():
            for m in self.model.modules():
                if isinstance(m, BatchNorm2d):
                    nn.init.ones_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                    m.running_mean.zero_()
                    m.running_var.fill_(1)
                    m.num_batches_tracked.zero_()
                if isinstance(m, Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if isinstance(m, Linear):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, inputs, targets):
        out = self.model.forward(inputs)
        out = self.locater(out, targets)
        loss = self.criterion(out, targets)

        weights = torch.cat([m.weight.view(-1) for m in self.model.modules() if isinstance(m, Conv2d)])
        l2_loss = 1./math.sqrt(weights.numel()) * torch.norm(weights, 2)
        loss += self.args.lam * l2_loss

        self.optimizer.zero_grad()
        loss.backward()
        return loss.detach(), out
    
    def train(self, epoch_idx):
        self.model.train()
        train_loader = self.train_loader
        train_loss = Metric('train_loss')
        train_accuracy = Metric('train_accuracy')

        with tqdm(total=len(train_loader),
                  desc='Train Ep. #{}: '.format(epoch_idx + 1),
                  disable=False,
                  ascii=True) as t:
            for batch_idx, (input, target) in enumerate(train_loader):
                input, target = input.cuda(), target.cuda()
                loss, output = self.forward(input, target)
        
                self.optimizer.step()
        
                num = input.size(0)
                train_accuracy.update(self.classification_accuracy(output, target), num)
                train_loss.update(loss.cpu(), num)

                t.set_postfix({
                    'lr': '{:.2f}'.format(self.optimizer.param_groups[0]['lr']),
                    'loss': '{:.2f}'.format(train_loss.avg.item()),
                    'acc': '{:.2f}'.format(100. * train_accuracy.avg.item()),
                    })
                t.update(1)
        if epoch_idx >= 150:
            self.scheduler.step()
        summary = {
            'loss': train_loss.avg.item(),
            'acc': 100. * train_accuracy.avg.item()
            }
        self.validate()
        return summary

    def validate(self):
        self.model.eval()
        val_loader = self.val_loader
        loss = Metric('loss')
        accuracy = Metric('accuracy')

        self.count = {}
        for i in range(self.labels.shape[0]):
            self.count[i] = 0

        with tqdm(total=len(val_loader),
                  desc='Val: ',
                  ascii=True) as t:
            with torch.no_grad():
                sum = 0
                for (data, target) in val_loader:
                    data, target = data.cuda(), target.cuda()

                    output = self.model(data)
                    output = self.locater(output, target)
                    num = data.size(0)
                    sum += num
                    self.correct_count(output, target)
                    loss.update(self.criterion(output, target).cpu(), num)
                    accuracy.update(self.classification_accuracy(output, target), num)

                    t.set_postfix({ 
                        'loss': '{:.2f}'.format(loss.avg.item()),
                        'acc': '{:.2f}'.format(100. * accuracy.avg.item()),
                        })
                    t.update(1)

        accuracy_ = {}
        for k, v in self.count.items():
            accuracy_[k] = v * (100//self.args.num_classes)/sum

        summary = {
            'loss': loss.avg.item(),
            'acc': 100. * accuracy.avg.item(),
            }
        return summary, accuracy_
    
    def locater(self, out, target):
        out, target = out.cpu(), target.cpu()
        output = torch.zeros(out.shape)
        for i in range(out.shape[0]):
            for cls in self.labels:
                if target[i] in cls:
                    output[i][cls] = out[i][cls]
        return output.cuda()
        
    def classification_accuracy(self, output, target):
        output, target = output.cpu(), target.cpu()
        pred = torch.empty((output.shape[0], 1))
        for i in range(output.shape[0]):
            for cls in self.labels:
                if target[i] in cls:
                    pred[i][0] = cls[output[i][cls].max(0, keepdim=True)[1]]

        return pred.eq(target.view_as(pred)).float().mean()

    def correct_count(self, output, target):
        output, target = output.cpu(), target.cpu()
        
        for i in range(output.shape[0]):
            for j in range(self.labels.shape[0]):
                cls = self.labels[j]
                if target[i] in cls:
                    pred = cls[output[i][cls].max(0, keepdim=True)[1]]
                    if pred == target[i]:
                        self.count[j] += 1 

        