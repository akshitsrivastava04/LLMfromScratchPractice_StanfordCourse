import unittest
import torch
import sys
import os

# Add the parent directory to sys.path to allow imports from Layers and Attention
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Transformer.transformer import TransformerBlock

class TestTransformerBlock(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "emb_dim": 64,
            "num_heads": 4,
            "drop_rate": 0.1,
            "qkv_bias": True,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "hidden_dim": 128,
            "context_length": 1024,
            "enable_profiling": False
        }
        self.batch_size = 2
        self.seq_len = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_init(self):
        """Test initialization of TransformerBlock"""
        block = TransformerBlock(self.cfg).to(self.device)
        self.assertIsInstance(block, TransformerBlock)
        self.assertTrue(hasattr(block, 'att'))
        self.assertTrue(hasattr(block, 'ff'))
        self.assertTrue(hasattr(block, 'norm1'))
        self.assertTrue(hasattr(block, 'norm2'))

    def test_forward_pass(self):
        """Test forward pass output shape and type"""
        block = TransformerBlock(self.cfg).to(self.device)
        x = torch.randn(self.batch_size, self.seq_len, self.cfg["emb_dim"]).to(self.device)
        
        # Test without aux loss
        output = block(x, return_aux_loss=False)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.cfg["emb_dim"]))

    def test_forward_pass_with_aux_loss(self):
        """Test forward pass with auxiliary loss"""
        block = TransformerBlock(self.cfg).to(self.device)
        x = torch.randn(self.batch_size, self.seq_len, self.cfg["emb_dim"]).to(self.device)
        
        # Enable training mode for aux loss
        block.train()
        output, aux_loss = block(x, return_aux_loss=True)
        
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.cfg["emb_dim"]))
        # aux_loss might be None if not training or specific conditions, but with MoE it usually returns a value
        if aux_loss is not None:
            self.assertIsInstance(aux_loss, torch.Tensor)

    def test_profiling(self):
        """Test profiling functionality"""
        cfg_profiling = self.cfg.copy()
        cfg_profiling["enable_profiling"] = True
        block = TransformerBlock(cfg_profiling).to(self.device)
        
        x = torch.randn(self.batch_size, self.seq_len, self.cfg["emb_dim"]).to(self.device)
        block(x)
        
        stats = block.get_profiling_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("time_ms", stats)
        self.assertIn("memory_bytes", stats)
        self.assertTrue(len(stats["time_ms"]) > 0)

    def test_dense_feedforward(self):
        """Test with dense FeedForward (num_experts=0)"""
        cfg_dense = self.cfg.copy()
        cfg_dense["num_experts"] = 0
        block = TransformerBlock(cfg_dense).to(self.device)
        
        x = torch.randn(self.batch_size, self.seq_len, self.cfg["emb_dim"]).to(self.device)
        output = block(x, return_aux_loss=False)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.cfg["emb_dim"]))
        # Dense FF shouldn't return aux loss usually, or it should be None
        output_with_aux = block(x, return_aux_loss=True)
        if isinstance(output_with_aux, tuple):
             _, aux_loss = output_with_aux
             self.assertIsNone(aux_loss)

if __name__ == '__main__':
    unittest.main()
