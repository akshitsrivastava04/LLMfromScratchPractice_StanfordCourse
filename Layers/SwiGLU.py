import torch 
import torch.nn as nn 
import torch.nn.functional as F
from Attention.SelfAttention import Linear, Dropout

class SwiGLU(nn.Module):
     """
     SwiGLU activation function 
     
     eq: swish(x) = x * sigmoid(beta * x) 
     """
     def __init__(self, dim, hidden_dim=None, bias=True, dropout=0.0): 
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim 
        self.w = Linear(dim, 2 * hidden_dim, bias=bias)
        self.hidden_dim = hidden_dim 

     def forward(self, x):
          x_proj = self.w(x)
          x_gate, x_value = x_proj.split(self.hidden_dim, dim=-1)
          return  F.silu(x_gate) * x_value

     