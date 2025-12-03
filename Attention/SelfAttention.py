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

class CasualAttention(nn.Module): 
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out 
        self.W_query = Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = Dropout(dropout)
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).to(torch.bool)
        self.register_buffer("mask", mask)
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape 
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = torch.einsum("bqd, bkd -> bqk", queries, keys)
        attn_scores.masked_fill_(self.mask[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        self.attn_weights = attn_weights
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

    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    values = sa_v2.W_value(inputs)
    
    print("\nQueries shape:", queries.shape)
    print("Keys shape:", keys.shape)
    print("Values shape:", values.shape)

    attn_scores = torch.einsum("bqd, bkd -> bqk", queries, keys)
    print("\nAttention scores shape:", attn_scores.shape)
    attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
    print("\nAttention weights shape:", attn_weights.shape)
    context_vec = torch.einsum('bqk,bkv->bqv', attn_weights, values)
    print("\nContext vector shape:", context_vec.shape)
    context_lenght = attn_scores.shape[0]
    mask_simple = torch.tril(torch.ones(context_lenght, context_lenght)) 
    print(mask_simple)

    print("\n--- Testing CasualAttention ---")
    context_len = 5
    ca = CasualAttention(d_in, d_out, context_length=context_len, dropout=0.1, qkv_bias=True)

    x = torch.randn(2, 5, d_in)   # (batch, seq_len, dim)
    y2 = ca(x)

    print("CasualAttention Output shape:", y2.shape)

    loss2 = y2.sum()
    loss2.backward()

    print("Grad W_query:", ca.W_query.W.grad is not None)
    print("Mask used:\n", ca.mask[:5, :5])