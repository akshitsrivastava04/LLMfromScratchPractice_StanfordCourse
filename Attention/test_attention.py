import unittest
import torch
import torch.nn as nn
import time
from SelfAttention import Linear, SelfAttention_v1, SelfAttention_v2, CasualAttention

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

    def test_casual_attention_forward(self):
        print("\nTesting CasualAttention:")
        # Use a larger context length than the sequence length to ensure mask works
        context_len = 5
        ca = CasualAttention(self.d_in, self.d_out, context_length=context_len, dropout=0.0)
        
        start_time = time.time()
        output = ca(self.input_tensor)
        end_time = time.time()
        
        print(f"  Input shape: {self.input_tensor.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output values:\n{output}")
        print(f"  Attention Weights:\n{ca.attn_weights}")
        print(f"  Execution time: {end_time - start_time:.6f} seconds")
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_out))
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN")
        
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(ca.W_query.W.grad)
        self.assertIsNotNone(ca.W_key.W.grad)
        self.assertIsNotNone(ca.W_value.W.grad)
        print("  Gradients computed successfully.")

    def test_casual_attention_masking(self):
        print("\nTesting CasualAttention Masking Property:")
        # Verify that future tokens do not affect past tokens
        context_len = 5
        # Set dropout to 0 for deterministic output
        ca = CasualAttention(self.d_in, self.d_out, context_length=context_len, dropout=0.0)
        
        # Create input
        input1 = self.input_tensor.clone()
        
        # Create a second input where we change the last token of the sequence
        input2 = self.input_tensor.clone()
        input2[:, -1, :] = torch.randn_like(input2[:, -1, :]) # Change the last token (t=2)
        
        # Get outputs
        output1 = ca(input1)
        output2 = ca(input2)
        
        # Check that outputs at t=0 and t=1 are IDENTICAL
        # The change at t=2 should NOT affect t=0 and t=1
        
        # We check the first two tokens (indices 0 and 1)
        diff = (output1[:, :2, :] - output2[:, :2, :]).abs().max()
        print(f"  Max difference for past tokens when future token is changed: {diff.item()}")
        
        self.assertTrue(torch.allclose(output1[:, :2, :], output2[:, :2, :], atol=1e-6), 
                        "Output at t < T should not change when input at T changes")
        
        # Check that output at t=2 IS different (since input at t=2 changed)
        # Note: It's theoretically possible they are same if weights are zero, but highly unlikely with random init
        diff_last = (output1[:, -1, :] - output2[:, -1, :]).abs().max()
        print(f"  Max difference for current token when current token is changed: {diff_last.item()}")
        self.assertFalse(torch.allclose(output1[:, -1, :], output2[:, -1, :]),
                         "Output at t=T should change when input at T changes")
        print("  Causal masking verified.")

    def test_dropout_effect(self):
        print("\nTesting Dropout Effect:")
        context_len = 5
        ca = CasualAttention(self.d_in, self.d_out, context_length=context_len, dropout=0.5)
        
        # CRITICAL: Set to training mode
        ca.train()
        
        torch.manual_seed(42)
        x = torch.randn(1, 3, self.d_in)
        
        # Run twice with same input - should get different outputs due to dropout
        output1 = ca(x)
        output2 = ca(x)
        
        print(f"  Training mode - Output 1 (first 3 values): {output1[0, 0, :3]}")
        print(f"  Training mode - Output 2 (first 3 values): {output2[0, 0, :3]}")
        print(f"  Outputs different (dropout active): {not torch.allclose(output1, output2)}")
        
        self.assertFalse(torch.allclose(output1, output2), 
                        "Outputs should differ in training mode with dropout")
        
        # Now test eval mode
        ca.eval()
        output3 = ca(x)
        output4 = ca(x)
        
        print(f"  Eval mode - Output 3 (first 3 values): {output3[0, 0, :3]}")
        print(f"  Eval mode - Output 4 (first 3 values): {output4[0, 0, :3]}")
        print(f"  Outputs identical (dropout disabled): {torch.allclose(output3, output4)}")
        
        self.assertTrue(torch.allclose(output3, output4), 
                        "Outputs should be identical in eval mode")
        
        print("  Dropout functionality verified.")

if __name__ == '__main__':
    unittest.main()
