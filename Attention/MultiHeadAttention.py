import torch 
from SelfAttention import Linear, Dropout
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module): 
    def __init__(self, d_in, d_out, context_length, dropout, num_heads,qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        
        self.d_out = d_out 
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.context_length = context_length

        self.W_query = Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = Dropout(dropout)
        
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        self.register_buffer("mask", mask)
        self.out_proj = Linear(d_out, d_out)


        #-----KV Cache-----
        self.register_buffer("cache_k", torch.empty(0), persistent=False)
        self.register_buffer("cache_v", torch.empty(0), persistent=False)
        self.ptr_current_pos = 0

    def _init_cache(self, batch_size, device, dtype):
        self.cache_k = torch.zeros(batch_size, self.num_heads, self.context_length, self.head_dim, device=device, dtype=dtype)
        self.cache_v = torch.zeros(batch_size, self.num_heads, self.context_length, self.head_dim, device=device, dtype=dtype)
        self.ptr_current_pos = 0

    def forward(self,x):
        b, num_tokens, d_in = x.shape
        assert num_tokens <= self.mask.shape[0], "Input sequence exceeds maximum context length"
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        #--- Handling KV Caching ---

        if use_cache:
            if self.cache_k.numel() == 0 or self.cache_k.shape[0] != b:
                self._init_cache(batch_size=b, device=x.device, dtype=keys.dtype)

            assert self.ptr_current_pos + num_tokens <= self.context_length, (
                f"Cache overflow: ptr {self.ptr_current_pos} + tokens {num_tokens} > context_length {self.context_length}"
            )

            self.cache_k[:, :, self.ptr_current_pos: self.ptr_current_pos + num_tokens, :] = keys 
            self.cache_v[:, :, self.ptr_current_post: self.ptr_current_pos + num_tokens, :] = values 

            self.ptr_current_pos += num_tokens
            keys = self.cache_k[:, :self.ptr_current_pos, :]
            values = self.cache_v[:, :, :self.ptr_current_pos, :]
            num_k = self.ptr_current_pos
            num_q = num_tokens
        else: 
            keys = keys
            values = values 
            num_k = num_q = num_tokens 
            self.ptr_current_pos = 0 
            
        attn_scores = torch.einsum("bnqd,bnkd->bnqk", queries, keys)

        mask_bool = self.mask[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores /  keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = torch.einsum("bnqk,bnkd->bnqd", attn_weights, values)
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec 

    def reset_cache(self):
        self.cache_k = torch.empty(0)
        self.cache_v = torch.empty(0)
        self.ptr_current_pos = 0

if __name__ == "__main__":
    print("="*60)
    print("Testing MultiHeadAttention Module")
    print("="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Hyperparameters
    batch_size = 2
    context_length = 10
    d_in = 512
    d_out = 512
    num_heads = 8
    dropout = 0.1
    
    print(f"\nHyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Context length: {context_length}")
    print(f"  d_in: {d_in}")
    print(f"  d_out: {d_out}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Dropout: {dropout}")
    
    # Create random input
    x = torch.randn(batch_size, context_length, d_in)
    print(f"\nInput shape: {x.shape}")
    print(f"Input sample (first 3 values of first token):\n  {x[0, 0, :3]}")
    
    # Initialize MultiHeadAttention
    mha = MultiHeadAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=dropout,
        num_heads=num_heads,
        qkv_bias=False
    )
    
    print(f"\n" + "="*60)
    print("Model Architecture:")
    print("="*60)
    print(mha)
    
    # Count parameters
    total_params = sum(p.numel() for p in mha.parameters())
    trainable_params = sum(p.numel() for p in mha.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Set to evaluation mode to disable dropout
    mha.eval()
    
    print(f"\n" + "="*60)
    print("Forward Pass:")
    print("="*60)
    
    # Forward pass
    with torch.no_grad():
        output = mha(x)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output sample (first 3 values of first token):\n  {output[0, 0, :3]}")
    
    # Check output statistics
    print(f"\nOutput Statistics:")
    print(f"  Mean: {output.mean().item():.6f}")
    print(f"  Std: {output.std().item():.6f}")
    print(f"  Min: {output.min().item():.6f}")
    print(f"  Max: {output.max().item():.6f}")
    
    # Test mask
    print(f"\n" + "="*60)
    print("Mask Visualization:")
    print("="*60)
    print(f"Mask shape: {mha.mask.shape}")
    print(f"Maximum allowed sequence length: {mha.mask.shape[0]}")
    print(f"\nMask (1 = masked, 0 = not masked):")
    print(mha.mask[:context_length, :context_length].int())
    
    # Test with different sequence lengths
    print(f"\n" + "="*60)
    print("Testing with Different Sequence Lengths:")
    print("="*60)
    
    for seq_len in [3, 5, 10]:
        x_test = torch.randn(1, seq_len, d_in)
        with torch.no_grad():
            output_test = mha(x_test)
        print(f"\nSequence length: {seq_len}")
        print(f"  Input shape: {x_test.shape}")
        print(f"  Output shape: {output_test.shape}")
        print(f"  Output mean: {output_test.mean().item():.6f}")
        print(f"  ✓ Passed assertion check")
    
    # Test assertion error (sequence too long)
    print(f"\n" + "="*60)
    print("Testing Assertion Error (Sequence Too Long):")
    print("="*60)
    
    try:
        seq_len_too_long = context_length + 5
        x_too_long = torch.randn(1, seq_len_too_long, d_in)
        print(f"\nTrying with sequence length: {seq_len_too_long} (max allowed: {context_length})")
        with torch.no_grad():
            output_fail = mha(x_too_long)
        print("  ✗ ERROR: Should have raised an assertion error!")
    except AssertionError as e:
        print(f"  ✓ Correctly caught assertion error:")
        print(f"    '{e}'")
    
    # Test gradient flow
    print(f"\n" + "="*60)
    print("Testing Gradient Flow:")
    print("="*60)
    
    mha.train()  # Set to training mode
    x_grad = torch.randn(batch_size, context_length, d_in, requires_grad=True)
    output_grad = mha(x_grad)
    loss = output_grad.sum()
    loss.backward()
    
    print(f"\nGradients computed successfully!")
    print(f"  Input gradient shape: {x_grad.grad.shape}")
    print(f"  Input gradient mean: {x_grad.grad.mean().item():.6f}")
    
    # Access custom Linear layer weights (using .W instead of .weight)
    print(f"\nWeight Matrix Gradients:")
    print(f"  W_query.W gradient mean: {mha.W_query.W.grad.mean().item():.6f}")
    print(f"  W_key.W gradient mean: {mha.W_key.W.grad.mean().item():.6f}")
    print(f"  W_value.W gradient mean: {mha.W_value.W.grad.mean().item():.6f}")
    print(f"  out_proj.W gradient mean: {mha.out_proj.W.grad.mean().item():.6f}")
    
    if mha.out_proj.b is not None:
        print(f"  out_proj.b gradient mean: {mha.out_proj.b.grad.mean().item():.6f}")
    
    # Detailed parameter gradients
    print(f"\nAll Parameter Gradients:")
    for name, param in mha.named_parameters():
        if param.grad is not None:
            print(f"  {name}:")
            print(f"    Shape: {param.grad.shape}")
            print(f"    Mean: {param.grad.mean().item():.6f}")
            print(f"    Std: {param.grad.std().item():.6f}")
            print(f"    Min: {param.grad.min().item():.6f}")
            print(f"    Max: {param.grad.max().item():.6f}")
    
    # Test attention head dimensions
    print(f"\n" + "="*60)
    print("Attention Head Dimensions:")
    print("="*60)
    print(f"  Head dimension: {mha.head_dim}")
    print(f"  Number of heads: {mha.num_heads}")
    print(f"  Total d_out: {mha.d_out}")
    print(f"  Verification: {mha.num_heads} × {mha.head_dim} = {mha.num_heads * mha.head_dim}")
    
    # Test edge case: sequence length equals context length
    print(f"\n" + "="*60)
    print("Testing Edge Case (Max Sequence Length):")
    print("="*60)
    
    x_max = torch.randn(1, context_length, d_in)
    with torch.no_grad():
        output_max = mha(x_max)
    print(f"\nSequence length = Context length = {context_length}")
    print(f"  Input shape: {x_max.shape}")
    print(f"  Output shape: {output_max.shape}")
    print(f"  ✓ Passed assertion check")
    
    # Test weight shapes
    print(f"\n" + "="*60)
    print("Weight Matrix Shapes:")
    print("="*60)
    print(f"  W_query.W: {mha.W_query.W.shape}")
    print(f"  W_key.W: {mha.W_key.W.shape}")
    print(f"  W_value.W: {mha.W_value.W.shape}")
    print(f"  out_proj.W: {mha.out_proj.W.shape}")
    if mha.out_proj.b is not None:
        print(f"  out_proj.b: {mha.out_proj.b.shape}")
    
    print(f"\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
