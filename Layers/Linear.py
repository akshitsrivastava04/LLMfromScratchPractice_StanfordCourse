import torch.nn as nn 
import torch
import torch.nn.functional as F
import math


class Linear(nn.Module):
    def __init__(self, d_in, d_out, bias=True):
        super().__init__()
        
        W = torch.empty(d_in, d_out) 
        W = nn.Parameter(W)   
        self.W = W

        if bias: 
            b_tensor = torch.empty(d_out, requires_grad=True)
            self.register_parameter("b", nn.Parameter(b_tensor))
        else: 
            self.b = None 
        
        self.reset_parameters()

    def reset_parameters(self): 
        fan_in = self.W.size(0)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.b is not None: 
            torch.nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x): 
        y = torch.einsum("bsi,io->bso", x, self.W)

        if self.b is not None: 
            y = y + self.b
        return y 