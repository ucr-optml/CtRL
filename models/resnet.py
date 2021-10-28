import math
from re import S
import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from . import Cell
import torch.nn.functional as F


__all__ = ['GEMResNet18', 'ResNet50', 'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']



class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        dilation: int = 1,
    ) -> None:
        super(BasicBlock, self).__init__()
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.cell1 = Cell(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.cell2 = Cell(planes, planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.cell1(x)
        out = self.relu(out)
        out = self.cell2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def update_input_masks(self, input_mask):
        self.cell1.mask &= input_mask.view(1, -1, 1, 1)
        self.cell2.mask &= self.cell1.get_output_mask().view(1, -1, 1, 1)
        if self.downsample is not None:
            self.downsample.mask &= input_mask.view(1, -1, 1, 1)
            return self.downsample.get_output_mask() | self.cell2.get_output_mask()
        else:
            return input_mask | self.cell2.get_output_mask()



class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.cell1 = Cell(inplanes, width)
        self.cell2 = Cell(width, width, stride, groups=groups, dilation=dilation)
        self.cell3 = Cell(width, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.cell1(x)
        out = self.relu(out)

        out = self.cell2(out)
        out = self.relu(out)

        out = self.cell3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def update_input_masks(self, input_mask):
        self.cell1.mask &= input_mask.view(1, -1, 1, 1)
        self.cell2.mask &= self.cell1.get_output_mask().view(1, -1, 1, 1)
        self.cell3.mask &= self.cell2.get_output_mask().view(1, -1, 1, 1)
        if self.downsample is not None:
            self.downsample.mask &= input_mask.view(1, -1, 1, 1)
            return self.downsample.get_output_mask() | self.cell3.get_output_mask()
        else:
            return input_mask | self.cell3.get_output_mask()



class GEMResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 20,
        replace_stride_with_dilation: Optional[List[bool]] = None,
    ) -> None:
        super(GEMResNet, self).__init__()

        self.inplanes = width_per_group
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.cell1 = Cell(3, self.inplanes, stride=1, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, width_per_group * 1, layers[0])
        self.layer2 = self._make_layer(block, width_per_group * 2, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, width_per_group * 4, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, width_per_group * 8, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.feature_num = width_per_group * 8 * block.expansion
        self.task_hist = {}
        self.classifiers = nn.ModuleDict()
        self.mask = None

    def _init_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, Cell):
                    nn.init.ones_(m.bn.weight)
                    if m.bn.bias is not None:
                        nn.init.zeros_(m.bn.bias)
                    m.bn.running_mean.zero_()
                    m.bn.running_var.fill_(1)
                    m.bn.num_batches_tracked.zero_()

                    m.conv.weight.data = m.conv.weight.data*~m.free_conv_mask + \
                        nn.init.kaiming_normal_(m.conv.weight, mode='fan_out', nonlinearity='relu').cuda() * m.free_conv_mask
                    
    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Cell(self.inplanes, planes * block.expansion, stride, kernel_size=1, padding=0)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                dilation=self.dilation))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, task_id) -> Tensor:
        # See note [TorchScript super()]
        x = self.cell1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if task_id is None:
            return x
        x = self.logits(x, task_id)
        return x

    def logits(self, x, task_id):
        x = self.classifiers[str(task_id)](x)
        return x

    def forward(self, x: Tensor, task_id=None) -> Tensor:
        return self._forward_impl(x, task_id)

    def update_masks(self):            
        input_mask = self.cell1.get_output_mask()

        for m in self.modules():
            if isinstance(m, (BasicBlock, Bottleneck)):
                input_mask=m.update_input_masks(input_mask)
        self.mask &= input_mask.view(1, -1)

        for m in self.modules():
            if isinstance(m, Cell):
                m.bn.weight.data *= m.get_output_mask()
                if m.bn.bias is not None:
                    m.bn.bias.data *= m.get_output_mask()
        # self.classifiers[str(task_id)].weight.data *= self.mask

    def add_task(self, task_id, num_classes):
        self.classifiers[str(task_id)] = nn.Linear(self.feature_num, num_classes).cuda()
        self.task_hist[str(task_id)] = {}
        self.task_hist[str(task_id)]['num_classes'] = num_classes
        self.task_hist[str(task_id)]['order'] = len(self.task_hist.keys())
        for m in self.modules():
            if isinstance(m, Cell):
                m.mask = torch.ones(m.conv.weight.shape, dtype=torch.bool).cuda()
        self.mask = torch.ones(self.classifiers[str(task_id)].weight.shape, dtype=torch.bool).cuda()
        self._init_weights()
        self.update_masks()

    def set_task(self, task_id, cp):
        if not str(task_id) in self.task_hist.keys():
            raise ValueError('task id {} is not trained!'.format(task_id))
        channel_weights = cp['channel_weights']
        channel_biases = cp['channel_biases']
        running_means = cp['running_means']
        running_vars = cp['running_vars']
        num_batches_tracked = cp['num_batches_tracked']
        masks = cp['masks']
        for n, m in self.named_modules():
            if isinstance(m, Cell):
                m.bn.weight.data = channel_weights[n]
                if m.bn.bias is not None:
                    m.bn.bias.data = channel_biases[n]
                m.bn.running_mean = running_means[n]
                m.bn.running_var = running_vars[n]
                m.bn.num_batches_tracked = num_batches_tracked[n]
                m.mask = masks[n]
        self.mask = torch.ones(self.classifiers[str(task_id)].weight.shape, dtype=torch.bool).cuda()
        self.update_masks()

    def load_state_dict(self, filename):
        checkpoint = torch.load(filename)
        task_hist = checkpoint['task_hist']
        state_dict = checkpoint['state_dict']
        free_masks = checkpoint['free_masks']
        self.classifiers = nn.ModuleDict({k: nn.Linear(self.feature_num, v['num_classes']).cuda() for k, v in task_hist.items()})
        super().load_state_dict(state_dict)
        self.task_hist = task_hist
        for n, m in self.named_modules():
            if isinstance(m, Cell):
                m.free_conv_mask = free_masks[n]
    
    def save_state_dict(self, filename):
        free_masks = {}
        for n, m in self.named_modules():
            if isinstance(m, Cell):
                free_masks[n] = m.free_conv_mask
        checkpoint = {}
        checkpoint['task_hist'] = self.task_hist
        checkpoint['state_dict'] = self.state_dict()
        checkpoint['free_masks'] = free_masks
        torch.save(checkpoint, filename)
    
    def task_exists(self, task_id):
        return str(task_id) in self.task_hist.keys()


## TODO
class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
    ) -> None:
        super(ResNet, self).__init__()

        self.inplanes = width_per_group
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.cell1 = Cell(3, self.inplanes, stride=2, kernel_size=7, padding=3)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, width_per_group * 1, layers[0])
        self.layer2 = self._make_layer(block, width_per_group * 2, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, width_per_group * 4, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, width_per_group * 8, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.last = nn.Linear(width_per_group * 8 * block.expansion, num_classes)
        self.last_input_mask = torch.ones(width_per_group * 8 * block.expansion, dtype=torch.bool).cuda()
        self.last_mask = torch.ones(self.last.weight.size(), dtype=torch.bool).cuda()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.cell3.bn.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.cell2.bn.weight, 0)  # type: ignore[arg-type]


    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Cell(self.inplanes, planes * block.expansion, stride, kernel_size=1, padding=0)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                dilation=self.dilation))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.cell1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.logits(x)
        return x

    def logits(self, x):
        w = self.last.weight * self.last_mask
        x = F.linear(x, w, self.last.bias)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def update_input_masks(self):            
        self.cell1.input_mask = torch.ones(3, dtype=torch.bool).cuda()
        input_mask = self.cell1.output_mask

        for m in self.layer1.modules():
            if isinstance(m, (BasicBlock, Bottleneck)):
                input_mask=m.update_input_masks(input_mask)

        for m in self.layer2.modules():
            if isinstance(m, (BasicBlock, Bottleneck)):
                input_mask=m.update_input_masks(input_mask)

        for m in self.layer3.modules():
            if isinstance(m, (BasicBlock, Bottleneck)):
                input_mask=m.update_input_masks(input_mask)

        for m in self.layer4.modules():
            if isinstance(m, (BasicBlock, Bottleneck)):
                input_mask=m.update_input_masks(input_mask)

        self.last_input_mask = input_mask.clone()  # [1,512,1,1]



def GEMResNet18(width_per_group=20) -> GEMResNet:
    return GEMResNet(BasicBlock, [2, 2, 2, 2], width_per_group=width_per_group)

################### TODO #######################
def ResNet50(num_classes) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)