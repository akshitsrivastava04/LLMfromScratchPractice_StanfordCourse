import unittest
import torch
from torch.utils.data import DataLoader
from Data_loader import GPTDatasetV1, create_dataloader

class MockTokenizer:
    def __init__(self):
        self.vocab_size = 100
        
    def encode(self, text):
        # Simple mock encoding: map each character to its unicode code point
        # This ensures deterministic output for testing
        return [ord(c) for c in text]

class TestGPTDatasetV1(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MockTokenizer()
        self.text = "Hello world! This is a test."
        self.max_length = 4
        self.stride = 2
        self.dataset = GPTDatasetV1(self.text, self.tokenizer, self.max_length, self.stride)

    def test_initialization(self):
        # Encoded: [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33, 32, 84, 104, 105, 115, 32, 105, 115, 32, 97, 32, 116, 101, 115, 116, 46]
        # Length of encoded text: 28
        # max_length: 4
        # stride: 2
        # Windows:
        # 0: [72, 101, 108, 108] -> Target: [101, 108, 108, 111]
        # 2: [108, 108, 111, 32] -> Target: [108, 111, 32, 119]
        # ...
        # Last start index: len - max_length = 24
        
        self.assertIsInstance(self.dataset.input_ids, list)
        self.assertIsInstance(self.dataset.target_ids, list)
        self.assertEqual(len(self.dataset.input_ids), len(self.dataset.target_ids))
        
        # Check first item
        input_0, target_0 = self.dataset[0]
        expected_input_0 = torch.tensor([72, 101, 108, 108])
        expected_target_0 = torch.tensor([101, 108, 108, 111])
        
        self.assertTrue(torch.equal(input_0, expected_input_0))
        self.assertTrue(torch.equal(target_0, expected_target_0))

    def test_len(self):
        # Total tokens: 28
        # max_length: 4
        # stride: 2
        # (28 - 4) // 2 + 1 = 13 chunks?
        # Let's verify loop: range(0, 24, 2) -> 0, 2, 4, ..., 22. 
        # 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22 -> 12 items.
        # Wait, range(0, 24, 2) excludes 24.
        # range(0, len(token_ids) - max_length, stride)
        # len(token_ids) - max_length = 28 - 4 = 24.
        # range(0, 24, 2) -> 0, 2, ..., 22. Count is 12.
        
        self.assertEqual(len(self.dataset), 12)

    def test_getitem(self):
        input_tensor, target_tensor = self.dataset[5]
        self.assertIsInstance(input_tensor, torch.Tensor)
        self.assertIsInstance(target_tensor, torch.Tensor)
        self.assertEqual(input_tensor.shape, (4,))
        self.assertEqual(target_tensor.shape, (4,))

class TestCreateDataLoader(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MockTokenizer()
        self.text = "".join([chr(i) for i in range(100)])
        self.batch_size = 2
        self.max_length = 4
        self.stride = 4
        
    def test_dataloader_creation(self):
        dataloader = create_dataloader(
            self.text, 
            self.tokenizer, 
            batch_size=self.batch_size, 
            max_length=self.max_length, 
            stride=self.stride,
            shuffle=False
        )
        self.assertIsInstance(dataloader, DataLoader)
        self.assertEqual(dataloader.batch_size, self.batch_size)
        
    def test_batch_dimensions(self):
        dataloader = create_dataloader(
            self.text, 
            self.tokenizer, 
            batch_size=self.batch_size, 
            max_length=self.max_length, 
            stride=self.stride,
            drop_last=True
        )
        
        # 100 tokens. max_len=4. range(0, 96, 4) -> 0, 4, ..., 92. 
        # 96 / 4 = 24 samples.
        # batch_size=2 -> 12 batches.
        
        print("\n--- Testing Batch Dimensions ---")
        batch_count = 0
        for inputs, targets in dataloader:
            batch_count += 1
            print(f"Batch {batch_count}: Inputs {inputs.shape}, Targets {targets.shape}")
            self.assertEqual(inputs.shape, (self.batch_size, self.max_length))
            self.assertEqual(targets.shape, (self.batch_size, self.max_length))
            
        self.assertEqual(batch_count, 12)

    def test_iteration_content(self):
        """Test iterating over the dataloader using iter() and next() to simulate training loop."""
        dataloader = create_dataloader(
            self.text, 
            self.tokenizer, 
            batch_size=self.batch_size, 
            max_length=self.max_length, 
            stride=self.stride,
            shuffle=False
        )
        
        data_iter = iter(dataloader)
        num_batches = 0
        
        # We expect 12 batches (calculated in test_batch_dimensions)
        expected_batches = 12
        
        print("\n--- Testing Iteration Content ---")
        for i in range(expected_batches):
            try:
                # Simulate fetching the next batch
                inputs, targets = next(data_iter)
                num_batches += 1
                
                print(f"Iter {i}: Inputs {inputs.shape}, Targets {targets.shape}")
                print(f"  Input[0]: {inputs[0].tolist()}")
                print(f"  Target[0]: {targets[0].tolist()}")

                # Verify batch structure
                self.assertIsInstance(inputs, torch.Tensor)
                self.assertIsInstance(targets, torch.Tensor)
                self.assertEqual(inputs.shape, (self.batch_size, self.max_length))
                self.assertEqual(targets.shape, (self.batch_size, self.max_length))
                
                # Verify that targets are shifted by 1 relative to inputs
                # This is a key property of the GPT dataset
                # inputs:  [t0, t1, t2, t3]
                # targets: [t1, t2, t3, t4]
                # So inputs[:, 1:] should equal targets[:, :-1]
                self.assertTrue(torch.equal(inputs[:, 1:], targets[:, :-1]))
                
            except StopIteration:
                self.fail(f"DataLoader stopped early at batch {i}")
        
        # Verify that the iterator is exhausted
        with self.assertRaises(StopIteration):
            next(data_iter)
            
        self.assertEqual(num_batches, expected_batches)

if __name__ == '__main__':
    unittest.main()
