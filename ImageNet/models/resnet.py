'''
resnet for cifar in pytorch
Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import torch
import torch.nn as nn
import numpy as np
# import math
# from models.quant_layer import QuantConv2d, first_conv, last_fc, QuantReLU
from models.quant_layer import QuantReLU

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                       padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Dummy(nn.Module):
    def __init__(self, block):
        super(Dummy, self).__init__()
        self.block = block
        self.idem = False
    def forward(self, x):
        if self.idem:
            return x
        return self.block(x)

class Dummy1(nn.Module):
    def __init__(self, block):
        super(Dummy1, self).__init__()
        self.block = block
        self.idem = False
    def forward(self, x):
        if self.idem:
            return x

        x = self.block[0](x)
        with open("ann.txt", "w") as file:
            file.write(f"annlayer0 output values: {x.detach().cpu().numpy()}\n")
        x = self.block[1](x)

        x = self.block[2](x)

        return x

class Oneway(nn.Module):
    def __init__(self, conv=None, bn=None, relu=None):
        super(Oneway, self).__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu 
        self.idem = False
    def forward(self, x):
        if self.idem:
            return x
        x = self.conv(x)
        x = self.bn(x) 
        x = self.relu(x)
        return x

class Twoways(nn.Module):
    def __init__(self, conv=None, bn=None, relu=None, downsample=None):
        super(Twoways, self).__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu 
        self.downsample = downsample
        self.idem = False
    def forward(self, x, identity):
        if self.idem:
            return x
        x = self.conv(x)
        x = self.bn(x) 
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
        return x   


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, bit=32):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.idem = False
        self.inter = False
        

        self.part1 = Oneway(conv3x3(inplanes, planes, stride),
                            norm_layer(planes),
                            QuantReLU(inplace=True, bit=bit)) 
        
        self.part2 = Twoways(conv3x3(planes, planes),
                             norm_layer(planes),
                             QuantReLU(inplace=True, bit=bit), downsample)            

    def forward(self, x):
        if self.idem:
            return x
        identity = x
        out = self.part1(x)
        if self.inter:
            return out
        out = self.part2(out, identity)
        return out
    
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, bit=32):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.bit = bit
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        # self.file = open("ann.txt", "w")
        self.layer0 = Dummy(nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                          nn.BatchNorm2d(64),
                                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                          QuantReLU(inplace=True, bit=bit)))
        #self.avg = Dummy(nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = Dummy(nn.Sequential(nn.AvgPool2d(7, stride=1),
                                                 nn.Flatten(1),
                                                 nn.Linear(512 * block.expansion, num_classes)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, bit=self.bit))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, bit=self.bit))

        return nn.Sequential(*layers)

        

    def forward(self, x):
        np.set_printoptions(precision=4)
        #self.file.write(f"x: {x.detach().cpu().numpy()}\n")
        x = self.layer0(x)

        # self.file.write(f"annlayer0 output values: {x.detach().cpu().numpy()}\n")
        #x = self.avg(x)
        x = self.layer1(x)  
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantReLU):
                m.show_params()


class Spiking(nn.Module):
    def __init__(self, block, T):
        super(Spiking, self).__init__()
        self.block = block
        self.T = T
        self.is_first = False
        self.idem = False
        self.sign = True
        
    def forward(self, x):
        if self.idem:
            return x
        
        ###initialize membrane to half threshold
        threshold = self.block[3].act_alpha.data*(2**(self.T-1))/(2**self.T-1)
        membrane = 0
        x = self.block(x)
        x.unsqueeze_(1)
        x = x.repeat(1, self.T, 1, 1, 1)

        
        #integrate charges
        
        for dt in range(self.T):
            membrane = x[:,dt]

        #membrane = self.block[3](membrane)
            
        for dt in range(self.T):
            if dt == 0:
                spike_train = torch.zeros(membrane.shape[:1] + torch.Size([self.T]) + membrane.shape[1:],device=membrane.device)
                
            spikes = membrane >= (1-0.5**(self.T-dt))*threshold
            membrane[spikes] = membrane[spikes] - threshold
            spikes = spikes.float()
            spike_train[:,dt] = spikes
            membrane *= 2
                
        spike_train = spike_train * threshold
        return spike_train


class Spiking_Oneway(nn.Module):
    def __init__(self, conv=None, bn=None, relu=None, T=0):
        super(Spiking_Oneway, self).__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu 
        self.idem = False
        self.T = T
        self.sign = True
    def forward(self, x):
        if self.idem:
            return x
        ###initialize membrane to half threshold
        threshold = self.relu.act_alpha.data*(2**(self.T-1))/(2**self.T-1)
        membrane = 0
        #sum_spikes = 0
        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)        
        x = self.conv(x)
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)

        x *= 2**x.size(1)-1
        x /= 2**(x.size(1)-1)
        x_list = []
        for i in range(x.size(1)):
            xi = x[:, i, :]
            xi = self.bn(xi)
            x_list.append(xi)

        x = torch.stack(x_list, dim=1)
        x /= 2**x.size(1)-1
        #prepare charges
        for dt in range(self.T):
            membrane = 2 * membrane + x[:,dt]
        #integrate charges
        for dt in range(self.T):
            if dt == 0:
                spike_train = torch.zeros(membrane.shape[:1] + torch.Size([self.T]) + membrane.shape[1:],device=membrane.device)
                
            spikes = membrane >= (1-0.5**(self.T-dt))*threshold
            membrane[spikes] = membrane[spikes] - threshold
            spikes = spikes.float()
            spike_train[:,dt] = spikes
            # 每个时间步后，膜电位变为原来的二分之一
            membrane *= 2                
        spike_train = spike_train * threshold
        return spike_train            
        
class Spiking_Twoways(nn.Module):
    def __init__(self, conv=None, bn=None, relu=None, downsample=None, T=0):
        super(Spiking_Twoways, self).__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu 
        self.downsample = downsample
        self.idem = False
        self.T = T
        self.sign = True
    def forward(self, x, identity):
        if self.idem:
            return x
        ###initialize membrane to half threshold
        threshold = self.relu.act_alpha.data*(2**(self.T-1))/(2**self.T-1)
        membrane = 0

        #prepare charges
        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.conv(x)
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)
        x *= 2**x.size(1)-1
        x /= 2**(x.size(1)-1)
        x_list = []
        for i in range(x.size(1)):
            xi = x[:, i, :]
            xi = self.bn(xi)
            x_list.append(xi)

        x = torch.stack(x_list, dim=1)
        x /= 2**x.size(1)-1
        
        x_sum = torch.zeros_like(x[:, 0, :])
        for i in range(x.size(1)):
            x_sum += (2**(x.size(1)-i-1)) * x[:, i, :]
        x = x_sum

        if self.downsample is not None:
            identity = self.downsample[0](identity)
            #print(identity.shape,1)
            identity = self.downsample[1](identity)
            #print(identity.shape,2)
            x += identity
        else:
            x += identity

        #integrate charges
        for dt in range(self.T):
            #membrane = 2 * membrane + x[:,dt]
            membrane = x

        for dt in range(self.T):
            if dt == 0:
                spike_train = torch.zeros(membrane.shape[:1] + torch.Size([self.T]) + membrane.shape[1:],device=membrane.device)
                
            spikes = membrane >= (1-0.5**(self.T-dt))*threshold
            membrane[spikes] = membrane[spikes] - threshold
            spikes = spikes.float()
            

            spike_train[:,dt] = spikes
            membrane *= 2
                
        spike_train = spike_train * threshold
        return spike_train   

        
class Spiking_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, bit=32):
        super(Spiking_BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.T = bit
        self.idem = False
        self.inter = False
        
        self.part1 = Spiking_Oneway(conv3x3(inplanes, planes, stride),
                                    norm_layer(planes),
                                    IF(), self.T) 
        
        self.part2 = Spiking_Twoways(conv3x3(planes, planes),
                                     norm_layer(planes),
                                     IF(), downsample, self.T)  

    def forward(self, x):
        if self.idem:
            return x
        x_sum = torch.zeros_like(x[:, 0, :])
        for i in range(x.size(1)):
            x_sum += (2**(x.size(1)-i-1)) * x[:, i, :]
        identity = x_sum / 2**(self.T-1)
        #identity = x
        out = self.part1(x)
        if self.inter:
            return out
        out = self.part2(out, identity)
        return out

'''class Avg_Spiking(nn.Module):
    def __init__(self, block, T):
        super(Avg_Spiking, self).__init__()
        self.block = block
        self.T = T
        self.idem = False
        
    def forward(self, x):
        if self.idem:
            return x
        #prepare charges
        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.block(x)
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)
        return x'''


class last_Spiking(nn.Module):
    def __init__(self, block, T):
        super(last_Spiking, self).__init__()
        self.block = block
        self.T = T
        self.idem = False
        
    def forward(self, x):
        if self.idem:
            return x
        #prepare charges
        x *= 2**x.size(1)-1
        x /= 2**(x.size(1)-1)
        x_list = []
        for i in range(x.size(1)):
            xi = x[:, i, :]
            xi = self.block(xi)
            x_list.append(xi)

        x = torch.stack(x_list, dim=1)
        x /= 2**x.size(1)-1

        
        #integrate charges
        # 积分电荷
        x_sum = torch.zeros_like(x[:, 0, :])
        for i in range(x.size(1)):
            x_sum += (2**(x.size(1)-i-1)) * x[:, i, :]
        x = x_sum / 2**(x.size(1)-1)

        #integrate charges
        return x

class IF(nn.Module):
    def __init__(self):
        super(IF, self).__init__()
        ###changes threshold to act_alpha
        ###being fleet
        self.act_alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.act_tau = torch.nn.Parameter(torch.tensor(2.0))
        self.act_tau_trans = torch.tensor(2.0) # placeholder

    def forward(self, x):
        return x
    
    def extra_repr(self) -> str:
        return 'threshold={:.3f}'.format(self.act_alpha)  


class S_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, bit=4):
        super(S_ResNet, self).__init__()
        self.inplanes = 64
        self.bit = bit
        self.T = bit
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        # self.file = open("snn.txt", "w")
        
        self.layer0 = Spiking(nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                          nn.BatchNorm2d(64),
                                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                          IF()), self.T)
        #self.avg = Avg_Spiking(nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)), self.T)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = last_Spiking(nn.Sequential(nn.AvgPool2d(7, stride=1),
                                                 nn.Flatten(1),
                                                 nn.Linear(512 * block.expansion, num_classes)), self.T)
        
        self.layer0.is_first = True

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, bit=self.bit))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, bit=self.bit))

        return nn.Sequential(*layers)

    def forward(self, x):
        #self.file.write(f"s: {x.detach().cpu().numpy()}\n")
        x = self.layer0(x)
        np.set_printoptions(precision=4)
        x = self.layer1(x)  
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantReLU):
                m.show_params()



def resnet18(spike=False, **kwargs):
    if spike:
        return S_ResNet(Spiking_BasicBlock, [3, 3 ,2], **kwargs)
    else:
        return ResNet(BasicBlock, [3, 3, 2], **kwargs)

def resnet32(spike=False, **kwargs):
    if spike:
        return S_ResNet(Spiking_BasicBlock, [5, 5, 5], **kwargs)
    else:
        return ResNet(BasicBlock, [5, 5, 5], **kwargs)
        
def resnet34(spike=False, **kwargs) -> ResNet:
    if spike:
        return S_ResNet(Spiking_BasicBlock, [3, 4, 6, 3], **kwargs)
    else:
        return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
 
    
def resnet44(spike=False, **kwargs):
    if spike:
        return S_ResNet(Spiking_BasicBlock, [7, 7, 7], **kwargs)
    else:
        return ResNet(BasicBlock, [7, 7, 7], **kwargs)


def resnet56(spike=False, **kwargs):
    if spike:
        return S_ResNet(Spiking_BasicBlock, [9, 9, 9], **kwargs)
    else:
        return ResNet(BasicBlock, [9, 9, 9], **kwargs)


def resnet110(spike=False, **kwargs):
    if spike:
        return S_ResNet(Spiking_BasicBlock, [18, 18, 18], **kwargs)
    else:
        return ResNet(BasicBlock, [18, 18, 18], **kwargs)


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    output_size = (input.size(2) + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    unfolded = F.unfold(input, kernel_size, dilation=dilation, padding=padding, stride=stride)
    pooled = unfolded.max(dim=1)[0]
    output = pooled.view(input.size(0), input.size(1), output_size, output_size)
    return output
