import torch
import torch.nn as nn
from models import *

def unsigned_spikes(model):
    for m in model.modules():
         if isinstance(m, Spiking):
             m.sign = False

#####the spiking wrapper######

class Spiking(nn.Module):
    def __init__(self, block, T):
        super(Spiking, self).__init__()
        self.block = block
        self.T = T
        self.is_first = False
        self.is_second = False
        self.is_third = False
        self.is_four = False
        self.is_five = False
        self.is_six = False
        self.idem = False
        self.sign = True 


    def forward(self, x):
        if self.idem:
            return x
        
        tau_trans = self.block[2].act_tau_trans.clamp(min=2.0, max=8.0)
        tau = self.block[2].act_tau.data.clamp(min=2.0, max=8.0)
        min_grid = self.block[2].act_alpha.data / ( (tau**self.T-1) / (tau -1) ) # => 0001  
        threshold = min_grid * (tau**(self.T-1)) # => 1000
        membrane = 0

        
        if self.is_first:
            # x.unsqueeze_(1)
            # x = x.repeat(1, self.T, 1, 1, 1)
            # train_shape = [x.shape[0], x.shape[1]]
            # x = x.flatten(0, 1)
            # x = self.block(x)
            # train_shape.extend(x.shape[1:])
            # x = x.reshape(train_shape)

            x = self.block(x)
            x.unsqueeze_(1)  
            x = x.repeat(1, self.T, 1, 1, 1)  
            membrane = x[:,0]


        else:

            x *= (tau_trans**x.size(1)-1) / (tau_trans - 1)

            train_shape = [x.shape[0], x.shape[1]]
            x = x.flatten(0, 1)

            x = self.block[0](x)
            x = self.block[1](x)
            
            train_shape.extend(x.shape[1:])
            x = x.reshape(train_shape)

            # x_list = []
            # for i in range(x.size(1)):
            #     xi = x[:, i, :]
            #     xi = self.block[1](xi)
            #     x_list.append(xi)
            # x = torch.stack(x_list, dim=1)

            x /= (tau_trans**x.size(1)-1) / (tau_trans - 1)
            x = self.block[2](x)

            #integrate charges
            for dt in range(self.T):
                membrane = tau_trans * membrane + x[:,dt]

        membrane = membrane + (0.5 * (1./tau)**(self.T-1) / (tau - 1)) * threshold

        for dt in range(self.T):

            if dt == 0:
                spike_train = torch.zeros(membrane.shape[:1] + torch.Size([self.T]) + membrane.shape[1:],device=membrane.device)

            # spikes = membrane >= (1-0.5**(self.T-dt))*threshold
            spikes = membrane > threshold * (0.5 * tau / (tau -1.))


            membrane[spikes] = membrane[spikes] - threshold
            spikes = spikes.float()
            
            spike_train[:,dt] = spikes
            membrane *= tau

        spike_train = spike_train * min_grid
        return spike_train

class last_Spiking(nn.Module):
    def __init__(self, block, T):
        super(last_Spiking, self).__init__()
        self.block = block
        self.T = T
        self.idem = False
        self.tau_trans = torch.tensor(2.0)
        
    def forward(self, x):
        if self.idem:
            return x
        #prepare charges
        tau_trans = self.tau_trans.clamp(min=2.0, max=8.0)
        x *= (tau_trans**x.size(1)-1) / (tau_trans - 1)

        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.block(x)   
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)

        x /= (tau_trans**x.size(1)-1) / (tau_trans - 1)

        
        #integrate charges
        x_sum = torch.zeros_like(x[:, 0, :])
        for i in range(x.size(1)):
           x_sum += (tau_trans**(x.size(1)-i-1)) * x[:, i, :]
        x = x_sum
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

    def show_params(self):
        act_alpha = round(self.act_alpha.data.item(), 3)
        print('clipping threshold activation alpha: {:2f}'.format(act_alpha)) 
    
    def extra_repr(self) -> str:
        return 'threshold={:.3f}'.format(self.act_alpha)  
