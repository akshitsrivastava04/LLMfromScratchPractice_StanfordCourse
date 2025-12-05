import torch.nn as nn 
import torch
import torch.nn.functional as F
import math

class Dropout(nn.Module):
    def __init__(self,p=0.5, inplace=False):
        super(Dropout, self).__init__()
        if p<0 or p>1:
            raise ValueError("Dropout probability must be between 0 and 1 but got {}".format(p))
        self.p = p 
        self.inplace = inplace 

    def forward(self, input):
        if self.training:
            mask = (torch.rand_like(input) > self.p).float()
            return mask * input / (1 - self.p)
        return input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) \
            + inplace_str + ')'