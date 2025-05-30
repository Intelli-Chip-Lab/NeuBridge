# Additive Power-of-Two Quantization: An Efficient Non-uniform Discretization For Neural Networks
# Yuhang Li, Xin Dong, Wei Wang
# International Conference on Learning Representations (ICLR), 2020.


import torch.nn as nn
import torch
import torch.nn.functional as F


def custom_round(x):
    return torch.floor(x + 0.5)

def build_power_value(B=2, additive=True):
    base_a = [0.]
    base_b = [0.]
    base_c = [0.]
    if additive:
        if B == 2:
            for i in range(3):
                base_a.append(2 ** (-i - 1))
        elif B == 4:
            for i in range(3):
                base_a.append(2 ** (-2 * i - 1))
                base_b.append(2 ** (-2 * i - 2))
        elif B == 6:
            for i in range(3):
                base_a.append(2 ** (-3 * i - 1))
                base_b.append(2 ** (-3 * i - 2))
                base_c.append(2 ** (-3 * i - 3))
        elif B == 3:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-i - 1))
                else:
                    base_b.append(2 ** (-i - 1))
                    base_a.append(2 ** (-i - 2))
        elif B == 5:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-2 * i - 1))
                    base_b.append(2 ** (-2 * i - 2))
                else:
                    base_c.append(2 ** (-2 * i - 1))
                    base_a.append(2 ** (-2 * i - 2))
                    base_b.append(2 ** (-2 * i - 3))
        else:
            pass
    else:
        for i in range(2 ** B - 1):
            base_a.append(2 ** (-i - 1))
    values = []
    for a in base_a:
        for b in base_b:
            for c in base_c:
                values.append((a + b + c))
    values = torch.Tensor(list(set(values)))
    values = values.mul(1.0 / torch.max(values))
    return values


def weight_quantization(b, grids, power=True):

    def uniform_quant(x, b):
        xdiv = x.mul((2 ** b - 1))
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    def power_quant(x, value_s):
        shape = x.shape
        xhard = x.view(-1)
        value_s = value_s.type_as(x)
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
        xhard = value_s[idxs].view(shape)
        return xhard

    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input.div_(alpha)
            input_c = input.clamp(min=-1, max=1)
            sign = input_c.sign()
            input_abs = input_c.abs()
            if power:
                input_q = power_quant(input_abs, grids).mul(sign)
            else:
                input_q = uniform_quant(input_abs, b).mul(sign)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            input, input_q = ctx.saved_tensors
            i = (input.abs()>1.).float()
            sign = input.sign()
            grad_alpha = (grad_output*(sign*i + (input_q-input)*(1-i))).sum()
            return grad_input, grad_alpha

    return _pq().apply


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit, power=True):
        super(weight_quantize_fn, self).__init__()
        self.w_bit = w_bit-1
        self.power = power if w_bit>2 else False
        self.grids = build_power_value(self.w_bit, additive=True)
        self.weight_q = weight_quantization(b=self.w_bit, grids=self.grids, power=self.power)
        self.register_parameter('wgt_alpha', Parameter(torch.tensor(3.0)))

    def forward(self, weight):
        if self.w_bit == 32:
            weight_q = weight
        else:
            mean = weight.data.mean()
            std = weight.data.std()
            weight = weight.add(-mean).div(std)
            weight_q = self.weight_q(weight, self.wgt_alpha)
        return weight_q


def act_quantization(b, grid, power=True):

    def uniform_quant(x, b=3):
        xdiv = x.mul(2 ** b - 1)
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    def power_quant(x, grid):
        shape = x.shape
        xhard = x.view(-1)
        value_s = grid.type_as(x)
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
        xhard = value_s[idxs].view(shape)
        return xhard

    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input=input.div(alpha)
            input_c = input.clamp(min=0, max=1)
            if power:
                input_q = power_quant(input_c, grid)
            else:
                input_q = uniform_quant(input_c, b)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            input, input_q = ctx.saved_tensors
            i = (input > 1.).float()
            grad_alpha = (grad_output * (i + (input_q - input) * (1 - i))).sum()
            grad_input = grad_input*(1-i)
            return grad_input, grad_alpha

    return _uq().apply


import math
def build_ap_grids(T=4, tau=2.0):

    # values = []
    values = torch.zeros((int(2**T))).cuda()
    for s in range(2**T):
        curr_value = 0
        for t in range(T):
            if math.floor(s * (0.5 ** t)) % 2 == 1:
                curr_value += tau**t
        # values.append(curr_value)
        values[s] = curr_value
    if tau == 1.0:
        upbound = T
    else:
        upbound = (tau**T - 1) / (tau - 1)
    values = values / upbound
    return values


def gradient_scale(x, scale):
    yout = x
    ygrad = x * scale
    y = (yout - ygrad).detach() + ygrad
    return y


def apot_quantization(tensor, alpha, proj_set, is_weight=True, grad_scale=None):
    def power_quant(x, value_s):
        if is_weight:
            shape = x.shape
            xhard = x.view(-1)
            sign = x.sign()
            value_s = value_s.type_as(x)
            xhard = xhard.abs()
            idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
            xhard = value_s[idxs].view(shape).mul(sign)
            xhard = xhard
        else:
            shape = x.shape
            xhard = x.view(-1)
            value_s = value_s.type_as(x)
            xhard = xhard
            idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
            xhard = value_s[idxs].view(shape)
            xhard = xhard
        xout = (xhard - x).detach() + x
        return xout

    if grad_scale:
        alpha = gradient_scale(alpha, grad_scale)
    data = tensor / alpha
    if is_weight:
        data = data.clamp(-1, 1)
        data_q = power_quant(data, proj_set)
        data_q = data_q * alpha
    else:
        data = data.clamp(0, 1)
        data_q = power_quant(data, proj_set)
        data_q = data_q * alpha
    return data_q


def uq_with_calibrated_graditens(grad_scale=None):
    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input.div_(alpha)
            input_c = input.clamp(min=-1, max=1)
            input_q = input_c.round()
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)  # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()  # calibration: grad for weights will not be clipped
            input, input_q = ctx.saved_tensors
            i = (input.abs() > 1.).float()
            sign = input.sign()
            grad_alpha = (grad_output * (sign * i + (input_q - input) * (1 - i))).sum()
            if grad_scale:
                grad_alpha = grad_alpha * grad_scale
            return grad_input, grad_alpha

    return _uq().apply


def uniform_quantization(tensor, alpha, bit, is_weight=True, grad_scale=None):
    if grad_scale:
        alpha = gradient_scale(alpha, grad_scale)
    data = tensor / alpha
    if is_weight:
        data = data.clamp(-1, 1)
        data = data * (2 ** (bit - 1) - 1)
        data_q = (data.round() - data).detach() + data
        data_q = data_q / (2 ** (bit - 1) - 1) * alpha
    else:
        data = data.clamp(0, 1)
        data = data * (2 ** bit - 1)
        data_q = (data.round() - data).detach() + data
        data_q = data_q / (2 ** bit - 1) * alpha
    return data_q


def act_quantization_AP(b):  

    def power_quant(x, grid):  
        x_backward = x
        x = x.detach()

        shape = x.shape  
        # print(shape)
        xhard = x.view(-1)  
        value_s = grid.type_as(x)  
        # value_s = value_s.detach()
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]  # 计算最近的量化级别的索引
        idxs = idxs.detach()
        
        xhard = torch.gather(value_s, dim=0, index=idxs.long()).view(shape)

        x_forward = xhard
        return x_forward + x_backward - x_backward.detach()  

    def act_func(x, alpha, tau):

        grid = build_ap_grids(T=b, tau=tau)
        x = x / alpha
        x = x.clamp(min=0, max=1)
        x = power_quant(x, grid)
        x = x * alpha
        return x 

    return act_func


#Convolution defauts to APoT weight quantizaiton 
class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, bit=5, power=True, additive=True, grad_scale=None):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias)
        self.layer_type = 'QuantConv2d'
        self.bit = bit
        self.power = power
        self.grad_scale = grad_scale
        if self.bit < 32:
            if power:
                if self.bit > 2:
                    self.proj_set_weight = build_power_value(B=self.bit - 1, additive=additive)
                self.proj_set_act = build_power_value(B=self.bit, additive=additive)
            self.weight_alpha = torch.nn.Parameter(torch.tensor(3.0))

    def forward(self, x):
        if self.bit == 32:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        # weight normalization
        mean = self.weight.mean()
        std = self.weight.std()
        weight = self.weight.add(-mean).div(std)
        if self.power:
            if self.bit > 2:
                weight = apot_quantization(weight, self.weight_alpha, self.proj_set_weight, True, self.grad_scale)
            else:
                weight = uq_with_calibrated_graditens(self.grad_scale)(weight, self.weight_alpha)
        else:
            if self.bit > 2:
                weight = uniform_quantization(weight, self.weight_alpha, self.bit, True, self.grad_scale)
            else:
                weight = uq_with_calibrated_graditens(self.grad_scale)(weight, self.weight_alpha)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def show_params(self):
        if self.bit != 32:
            wgt_alpha = round(self.weight_alpha.data.item(), 3)
            print('clipping threshold weight alpha: {:2f}'.format(wgt_alpha))


#ReLU defauts to uniform activation quantization
class QuantReLU(nn.ReLU):
    def __init__(self, inplace: bool = False, bit=5):
        super(QuantReLU, self).__init__(inplace)
        self.layer_type = 'QuantReLU'
        self.bit = bit
        # self.power = power
        self.act_alpha = torch.nn.Parameter(torch.tensor(2.0))
        self.act_tau = torch.nn.Parameter(torch.tensor(2.0))
        self.act_alq = act_quantization_AP(self.bit)
        self.tau_train = True  
    
    def tau(self):
        tau_forward = self.act_tau # range scope (2.0,8.0) (1.0,8.0)
        tau_backward = (self.act_tau) * 1.0 # gradient scale 0.5
        return tau_forward.detach() + tau_backward - tau_backward.detach()
        
    def forward(self, x):
        if self.bit == 32:
            return F.relu(x, inplace=self.inplace)
        # activaton quantization
        x = F.relu(x, inplace=self.inplace)
        self.act_tau.data = (self.act_tau).clamp(min=1.0, max=8.0)
        self.act_alpha.data = (self.act_alpha).clamp(min=0.1)
        if self.tau_train:
            return self.act_alq(x, self.act_alpha, self.tau())  
        else:
            return self.act_alq(x, self.act_alpha, self.tau().detach())  
        return x
        
    def show_params(self):
        if self.bit != 32:
            act_alpha = round(self.act_alpha.data.item(), 3)
            print('clipping threshold activation alpha: {:2f}'.format(act_alpha)) 
            print('clipping activation tau: {:2f}'.format((self.tau())))

#Linear defauts to APoT weight quantization
class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bit=5, power=True, additive=True, grad_scale=None):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.layer_type = 'QuantLinear'
        self.bit = bit
        self.power = power
        self.grad_scale = grad_scale
        if self.bit < 32:
            if power:
                if self.bit > 2:
                    self.proj_set_weight = build_power_value(B=self.bit - 1, additive=additive)
                self.proj_set_act = build_power_value(B=self.bit, additive=additive)
            self.weight_alpha = torch.nn.Parameter(torch.tensor(3.0))
        
    def forward(self, x):
        if self.bit == 32:
            return F.linear(x, self.weight, self.bias)
        # weight normalization
        mean = self.weight.mean()
        std = self.weight.std()
        weight = self.weight.add(-mean).div(std)
        if self.power:
            if self.bit > 2:
                weight = apot_quantization(weight, self.weight_alpha, self.proj_set_weight, True, self.grad_scale)
            else:
                weight = uq_with_calibrated_graditens(self.grad_scale)(weight, self.weight_alpha)
        else:
            if self.bit > 2:
                weight = uniform_quantization(weight, self.weight_alpha, self.bit, True, self.grad_scale)
            else:
                weight = uq_with_calibrated_graditens(self.grad_scale)(weight, self.weight_alpha)
        return F.linear(x, weight, self.bias)
    
    def show_params(self):
        if self.bit != 32:
            wgt_alpha = round(self.weight_alpha.data.item(), 3)
            print('clipping threshold weight alpha: {:.2f}'.format(wgt_alpha))



# 8-bit quantization for the first and the last layer
class first_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(first_conv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                         bias)
        self.layer_type = 'FConv2d'

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q - self.weight).detach() + self.weight
        return F.conv2d(x, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class last_fc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(last_fc, self).__init__(in_features, out_features, bias)
        self.layer_type = 'LFC'

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q - self.weight).detach() + self.weight
        return F.linear(x, weight_q, self.bias)
