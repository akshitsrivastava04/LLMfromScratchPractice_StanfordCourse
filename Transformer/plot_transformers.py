import torch
import time
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Transformer.transformer import TransformerBlock

def benchmark_transformer(seq_lens, emb_dims, num_experts=4, device='cpu'):
    results = {
        'time': {},
        'memory': {}
    }
    
    for dim in emb_dims:
        results['time'][dim] = []
        results['memory'][dim] = []
        
        cfg = {
            "emb_dim": dim,
            "num_heads": 4,
            "drop_rate": 0.0,
            "qkv_bias": True,
            "num_experts": num_experts,
            "num_experts_per_tok": 2,
            "hidden_dim": dim * 4,
            "context_length": 1024,
            "enable_profiling": True
        }
        
        block = TransformerBlock(cfg).to(device)
        block.eval()
        
        for seq_len in seq_lens:
            # Adjust batch size for memory constraints
            # Smaller batch for longer sequences or larger dimensions
            if device.type == 'cuda':
                if seq_len >= 512 or dim >= 768:
                    batch_size = 1
                elif seq_len >= 256 or dim >= 512:
                    batch_size = 2
                else:
                    batch_size = 4
            else:
                batch_size = 1
            
            x = torch.randn(batch_size, seq_len, dim).to(device)
            
            # Clear CUDA cache before benchmarking
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            try:
                # Warmup
                with torch.no_grad():
                    for _ in range(3):  # Reduced from 5 to save memory
                        block(x)
                
                # Reset stats
                block.reset_profiling_stats()
                
                # Benchmark
                with torch.no_grad():
                    for _ in range(10):  # Reduced from 20 to save memory
                        block(x)
                
                stats = block.get_profiling_stats()
                results['time'][dim].append(stats['avg_time_ms'])
                results['memory'][dim].append(stats['avg_memory_mb'])
                
                print(f"Dim: {dim}, Seq Len: {seq_len}, Batch: {batch_size}, "
                      f"Time: {stats['avg_time_ms']:.2f}ms, Mem: {stats['avg_memory_mb']:.2f}MB")
                
            except torch.cuda.OutOfMemoryError:
                print(f"⚠️  OOM at Dim: {dim}, Seq Len: {seq_len} - Skipping...")
                results['time'][dim].append(None)
                results['memory'][dim].append(None)
                
                # Clear cache and continue
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                continue
            
            except Exception as e:
                print(f"❌ Error at Dim: {dim}, Seq Len: {seq_len}: {e}")
                results['time'][dim].append(None)
                results['memory'][dim].append(None)
                continue
        
        # Clean up after each dimension
        del block
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return results

def plot_results(results, seq_lens, title_suffix=""):
    os.makedirs("Transformer/plots", exist_ok=True)
    
    # Plot Time
    plt.figure(figsize=(10, 6))
    for dim, times in results['time'].items():
        # Filter out None values
        valid_data = [(s, t) for s, t in zip(seq_lens, times) if t is not None]
        if valid_data:
            seq_lens_valid, times_valid = zip(*valid_data)
            plt.plot(seq_lens_valid, times_valid, marker='o', label=f'Emb Dim {dim}')
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Average Time (ms)')
    plt.title(f'TransformerBlock Execution Time {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Transformer/plots/time_benchmark{title_suffix.replace(' ', '_')}.png")
    plt.close()
    
    # Plot Memory
    plt.figure(figsize=(10, 6))
    for dim, mems in results['memory'].items():
        # Filter out None values
        valid_data = [(s, m) for s, m in zip(seq_lens, mems) if m is not None]
        if valid_data:
            seq_lens_valid, mems_valid = zip(*valid_data)
            plt.plot(seq_lens_valid, mems_valid, marker='o', label=f'Emb Dim {dim}')
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Average Memory (MB)')
    plt.title(f'TransformerBlock Memory Usage {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Transformer/plots/memory_benchmark{title_suffix.replace(' ', '_')}.png")
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Adjust test parameters based on available memory
    seq_lens = [128, 256, 512, 1024]
    emb_dims = [256, 512, 768]
    
    # For very limited GPU memory (< 6GB), reduce dimensions
    if device.type == 'cuda' and torch.cuda.get_device_properties(0).total_memory < 6e9:
        print("⚠️  Limited GPU memory detected - reducing test dimensions")
        emb_dims = [256, 512]
        seq_lens = [128, 256, 512]
    
    # Benchmark MoE
    print("\n" + "="*60)
    print("Benchmarking MoE TransformerBlock...")
    print("="*60)
    results_moe = benchmark_transformer(seq_lens, emb_dims, num_experts=4, device=device)
    plot_results(results_moe, seq_lens, title_suffix="(MoE)")
    
    # Benchmark Dense
    print("\n" + "="*60)
    print("Benchmarking Dense TransformerBlock...")
    print("="*60)
    results_dense = benchmark_transformer(seq_lens, emb_dims, num_experts=0, device=device)
    plot_results(results_dense, seq_lens, title_suffix="(Dense)")
    
    print("\n✓ Plots saved to Transformer/plots/")