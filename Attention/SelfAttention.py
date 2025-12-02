import torch.nn as nn 
import torch
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, d_in, d_out, bias=True):
        super().__init__()
        self.W = torch.randn(d_in, d_out) * (d_in ** -0.5)
        self.W.requires_grad = True 

        if bias: 
            self.b = torch.zeros(d_out)
            self.b.requires_grad = True 
        else: 
            self.b = None 

    def forward(self, x): 
        y = torch.einsum("bsi, io -> bso", x, self.W)
        if self.b is not None: 
            y = y + self.b
        return y 

class SelfAttention_v1(nn.Module): 
    def __init__(self, d_in, d_out): 
        super().__init__()
        self.d_out = d_out 
        self.W_query = torch.randn(d_in, d_out) * (d_in ** -0.5)
        self.W_query.requires_grad = True
        self.W_key = torch.randn(d_in, d_out) * (d_in ** -0.5)
        self.W_key.requires_grad = True
        self.W_value = torch.randn(d_in, d_out) * (d_in ** -0.5)
        self.W_value.requires_grad = True

    def forward(self, x): 
        # Einsum is a function that performs tensor operations using a concise string notation based on Einstien Summation Convention
        keys = torch.einsum("bsi, io -> bso", x, self.W_key)
        queries = torch.einsum("bsi, io -> bso", x, self.W_query)
        values = torch.einsum("bsi, io -> bso", x, self.W_value)
        
        attn_scores = torch.einsum("bqd, bkd -> bqk", queries, keys)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = torch.einsum('bqk,bkv->bqv', attn_weights, values)
        return context_vec

class SelfAttention_v2(nn.Module): 
    def __init__(self, d_in, d_out): 
        super().__init__()
        self.d_out = d_out 
        self.W_query = Linear(d_in, d_out, bias=False)
        self.W_key = Linear(d_in, d_out, bias=False)
        self.W_value = Linear(d_in, d_out, bias=False)

    def forward(self, x): 
        # Einsum is a function that performs tensor operations using a concise string notation based on Einstien Summation Convention
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = torch.einsum("bqd, bkd -> bqk", queries, keys)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = torch.einsum('bqk,bkv->bqv', attn_weights, values)
        return context_vec

if __name__ == "__main__":
    torch.manual_seed(123)

    d_in = 4
    d_out = 8
    

    sa_v2 = SelfAttention_v2(d_in, d_out)
    

    inputs = torch.randn(2, 3, d_in) 
    
    print("--- Running Test (SelfAttention_v2 with Manual Linear) ---")
    print(f"Input shape: {inputs.shape}")
    
    y = sa_v2(inputs)
    
    print(f"Output shape: {y.shape}")
    print("Output values:")
    print(y)
    

    loss = y.sum()
    loss.backward()

    print(f"\nGradient calculated for W_query: {sa_v2.W_query.W.grad is not None}")