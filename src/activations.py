import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU(inplace = True)
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    elif activation == 'elu':
        return nn.ELU(inplace=True)
    elif activation == 'swish':
        return Swish(inplace=True)  
    elif activation == 'mish':
        return Mish(inplace=True)
    elif activation == 'gelu':
        return nn.GELU()   
    else:
        raise NotImplementedError(f"Activation {activation} not implemented")