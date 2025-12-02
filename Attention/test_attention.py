import unittest
import torch
import torch.nn as nn
import time
from SelfAttention import Linear, SelfAttention_v1, SelfAttention_v2

class TestAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.d_in = 4
        self.d_out = 8
        self.batch_size = 2
        self.seq_len = 3
        self.input_tensor = torch.randn(self.batch_size, self.seq_len, self.d_in)
        print("\n" + "="*50)
        print(f"Test Setup: Batch Size={self.batch_size}, Seq Len={self.seq_len}, d_in={self.d_in}, d_out={self.d_out}")
        print("="*50)

    def test_linear_forward(self):
        print("\nTesting Linear Layer:")
        linear = Linear(self.d_in, self.d_out)
        
        start_time = time.time()
        output = linear(self.input_tensor)
        end_time = time.time()
        
        print(f"  Input shape: {self.input_tensor.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Execution time: {end_time - start_time:.6f} seconds")
        print(f"  Time Complexity: O(B * L * d_in * d_out)")
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_out))
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN")
        
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(linear.W.grad)
        self.assertIsNotNone(linear.b.grad)
        print("  Gradients computed successfully.")

    def test_linear_no_bias(self):
        print("\nTesting Linear Layer (No Bias):")
        linear = Linear(self.d_in, self.d_out, bias=False)
        
        start_time = time.time()
        output = linear(self.input_tensor)
        end_time = time.time()
        
        print(f"  Execution time: {end_time - start_time:.6f} seconds")
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_out))
        self.assertIsNone(linear.b)
        
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(linear.W.grad)

    def test_self_attention_v1_forward(self):
        print("\nTesting SelfAttention_v1:")
        sa = SelfAttention_v1(self.d_in, self.d_out)
        
        start_time = time.time()
        output = sa(self.input_tensor)
        end_time = time.time()
        
        print(f"  Input shape: {self.input_tensor.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output values:\n{output}")
        print(f"  Execution time: {end_time - start_time:.6f} seconds")
        print(f"  Time Complexity: O(L^2 * d_out) (Attention Matrix Calculation)")
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_out))
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN")

        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(sa.W_query.grad)
        self.assertIsNotNone(sa.W_key.grad)
        self.assertIsNotNone(sa.W_value.grad)
        print("  Gradients computed successfully.")

    def test_self_attention_v2_forward(self):
        print("\nTesting SelfAttention_v2:")
        sa = SelfAttention_v2(self.d_in, self.d_out)
        
        start_time = time.time()
        output = sa(self.input_tensor)
        end_time = time.time()
        
        print(f"  Input shape: {self.input_tensor.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output values:\n{output}")
        print(f"  Execution time: {end_time - start_time:.6f} seconds")
        print(f"  Time Complexity: O(L^2 * d_out) (Attention Matrix Calculation)")
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_out))
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN")

        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(sa.W_query.W.grad)
        self.assertIsNotNone(sa.W_key.W.grad)
        self.assertIsNotNone(sa.W_value.W.grad)
        print("  Gradients computed successfully.")

    def test_v1_v2_consistency_shapes(self):
        print("\nTesting Consistency between v1 and v2:")
        sa_v1 = SelfAttention_v1(self.d_in, self.d_out)
        sa_v2 = SelfAttention_v2(self.d_in, self.d_out)
        
        out_v1 = sa_v1(self.input_tensor)
        out_v2 = sa_v2(self.input_tensor)
        
        print(f"  v1 Output shape: {out_v1.shape}")
        print(f"  v2 Output shape: {out_v2.shape}")
        
        self.assertEqual(out_v1.shape, out_v2.shape)
        self.assertEqual(out_v1.dtype, out_v2.dtype)
        print("  Shapes and types match.")

if __name__ == '__main__':
    unittest.main()
