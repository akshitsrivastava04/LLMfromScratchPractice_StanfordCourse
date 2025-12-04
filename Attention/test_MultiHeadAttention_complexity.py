import torch
import time
import tracemalloc
import math
import sys
import os

# Add the current directory to sys.path to import MultiHeadAttention
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from MultiHeadAttention import MultiHeadAttention

def test_functional():
    print("="*60)
    print("Functional Tests")
    print("="*60)
    
    batch_size = 2
    context_length = 20
    d_in = 64
    d_out = 64
    num_heads = 4
    dropout = 0.0
    
    mha = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)
    mha.eval() # Disable dropout for deterministic output check if needed, though we just check shapes here
    
    # Test 1: Basic Forward Pass
    x = torch.randn(batch_size, context_length, d_in)
    try:
        output = mha(x)
        print(f"[PASS] Forward pass successful. Output shape: {output.shape}")
        assert output.shape == (batch_size, context_length, d_out), f"Expected {(batch_size, context_length, d_out)}, got {output.shape}"
    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")

    # Test 2: Variable Sequence Length
    seq_len = 10
    x_short = torch.randn(batch_size, seq_len, d_in)
    try:
        output_short = mha(x_short)
        print(f"[PASS] Variable sequence length ({seq_len}) successful. Output shape: {output_short.shape}")
        assert output_short.shape == (batch_size, seq_len, d_out)
    except Exception as e:
        print(f"[FAIL] Variable sequence length failed: {e}")

    # Test 3: d_out not divisible by num_heads
    try:
        MultiHeadAttention(d_in, 65, context_length, dropout, num_heads)
        print("[FAIL] Should have raised assertion error for d_out not divisible by num_heads")
    except AssertionError as e:
        print(f"[PASS] Correctly caught assertion error for invalid d_out/num_heads: {e}")
    except Exception as e:
        print(f"[FAIL] Caught unexpected exception: {e}")

def measure_time_complexity():
    print("\n" + "="*60)
    print("Time Complexity Analysis")
    print("="*60)
    
    d_in = 64
    d_out = 64
    num_heads = 4
    dropout = 0.0
    batch_size = 1
    
    # Sequence lengths to test
    seq_lengths = [128, 256, 512, 1024, 2048]
    times = []
    
    print(f"{'Seq Len':<10} | {'Time (ms)':<15} | {'Relative Increase':<20}")
    print("-" * 50)
    
    for i, seq_len in enumerate(seq_lengths):
        # Create model with large enough context length
        mha = MultiHeadAttention(d_in, d_out, seq_len, dropout, num_heads)
        mha.eval()
        x = torch.randn(batch_size, seq_len, d_in)
        
        # Warmup
        with torch.no_grad():
            _ = mha(x)
            
        # Measure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10): # Average over 10 runs
                _ = mha(x)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10 * 1000 # in ms
        times.append(avg_time)
        
        rel_increase = "N/A"
        if i > 0:
            ratio = avg_time / times[i-1]
            # Theoretical O(N^2) implies doubling N -> 4x time
            # But for smaller N, linear terms (projections) might dominate O(N)
            rel_increase = f"{ratio:.2f}x"
            
        print(f"{seq_len:<10} | {avg_time:<15.4f} | {rel_increase:<20}")

    print("\nNote: Attention mechanism theoretical time complexity is O(N^2 * d).")
    print("For small N, linear projections O(N * d^2) might dominate.")

def measure_space_complexity():
    print("\n" + "="*60)
    print("Space Complexity Analysis (Peak Memory)")
    print("="*60)
    
    d_in = 64
    d_out = 64
    num_heads = 4
    dropout = 0.0
    batch_size = 1
    
    seq_lengths = [128, 256, 512, 1024]
    
    print(f"{'Seq Len':<10} | {'Peak Memory (MB)':<20}")
    print("-" * 40)
    
    for seq_len in seq_lengths:
        tracemalloc.start()
        
        mha = MultiHeadAttention(d_in, d_out, seq_len, dropout, num_heads)
        mha.eval()
        x = torch.randn(batch_size, seq_len, d_in)
        
        with torch.no_grad():
            _ = mha(x)
            
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / 1024 / 1024
        print(f"{seq_len:<10} | {peak_mb:<20.4f}")

    print("\nNote: Attention mechanism theoretical space complexity is O(N^2) for the attention matrix.")

if __name__ == "__main__":
    test_functional()
    measure_time_complexity()
    measure_space_complexity()
