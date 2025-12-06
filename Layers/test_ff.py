import torch
import unittest
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

# Add parent directory to path to allow imports from Layers
# Assuming this script is run from the root directory or Layers directory
# We want to ensure we can import 'Layers' as a package if running from root
# But if we are in Layers, we might need to adjust.
# Best practice: Run from root: python Layers/test_ff.py

# Adjust path so we can import Layers
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Layers.MoEFeedForward import MoEFeedForward

class TestMoEFeedForward(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "emb_dim": 128,
            "hidden_dim": 512,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "load_balance_alpha": 0.01
        }
        self.model = MoEFeedForward(self.cfg)

    def test_initialization(self):
        """Test if the model initializes with correct shapes"""
        print("\nTesting Initialization...")
        self.assertEqual(self.model.w1.shape, (4, 128, 512))
        self.assertEqual(self.model.w2.shape, (4, 128, 512))
        self.assertEqual(self.model.w3.shape, (4, 512, 128))
        print("Initialization Test Passed!")

    def test_forward_pass(self):
        """Test forward pass output shape and auxiliary loss"""
        print("\nTesting Forward Pass...")
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, self.cfg["emb_dim"])
        
        output, aux_loss = self.model(x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, self.cfg["emb_dim"]))
        self.assertIsNotNone(aux_loss)
        print(f"Output Shape: {output.shape}")
        print(f"Aux Loss: {aux_loss.item()}")
        print("Forward Pass Test Passed!")

    def test_expert_usage(self):
        """Test if expert usage stats are being tracked"""
        print("\nTesting Expert Usage Stats...")
        # Need to be in training mode for stats to update
        self.model.train()
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, self.cfg["emb_dim"])
        
        # Reset stats first
        self.model.reset_expert_stats()
        
        # Forward pass
        _, _ = self.model(x)
        
        stats = self.model.get_expert_usage_stats()
        self.assertIsNotNone(stats)
        print("Expert Usage Stats:", stats)
        
        # Check if total tokens matches
        expected_tokens = batch_size * seq_len * self.cfg["num_experts_per_tok"]
        self.assertEqual(stats['total_tokens'], expected_tokens)
        print("Expert Usage Test Passed!")

def demo_and_plot():
    print("\nRunning Demo and Plotting...")
    cfg = {
        "emb_dim": 64,
        "hidden_dim": 256,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "load_balance_alpha": 0.01
    }
    model = MoEFeedForward(cfg)
    model.train() # Enable tracking
    
    # Create dummy data
    batch_size = 4
    seq_len = 32
    x = torch.randn(batch_size, seq_len, cfg["emb_dim"])
    
    # Forward pass
    output, loss = model(x)
    
    print(f"Demo Output Shape: {output.shape}")
    print(f"Demo Loss: {loss.item()}")
    
    # Get stats
    stats = model.get_expert_usage_stats()
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Output Activation Heatmap (first sequence in batch)
    plt.subplot(1, 2, 1)
    # Detach and convert to numpy
    output_np = output[0].detach().numpy()
    plt.imshow(output_np, aspect='auto', cmap='viridis')
    plt.title("Output Activations (Batch 0)")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Sequence Position")
    plt.colorbar()
    
    # Plot 2: Expert Usage
    plt.subplot(1, 2, 2)
    expert_counts = stats['expert_counts']
    experts = np.arange(cfg["num_experts"])
    plt.bar(experts, expert_counts)
    plt.title("Expert Usage Counts")
    plt.xlabel("Expert ID")
    plt.ylabel("Token Count")
    plt.xticks(experts)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(os.path.dirname(__file__), 'moe_ff_plot.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    # plt.show() # Commented out for non-interactive environments

if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Run demo and plot
    demo_and_plot()
