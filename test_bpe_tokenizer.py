import unittest
import os
import sys

# Add the current directory to sys.path so we can import the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from BPE_from_scratch import BPEtokenizerParams, train_bpe
except ImportError:
    # If the user hasn't moved the top-level code, this might trigger the download.
    # We can't easily prevent it without modifying their code, as per instructions.
    raise

class TestBPETokenizer(unittest.TestCase):
    
    def test_train_bpe_simple(self):
        """Test training on a simple repeated string."""
        text = "aaab"
        # We expect 'aa' to be merged if num_merges >= 1
        # 'a' is 97, 'b' is 98
        # 'aa' -> (97, 97) -> new token 256
        # text becomes [256, 97, 98]
        
        tokenizer = train_bpe(text, num_merges=1)
        
        self.assertIn(256, tokenizer.vocab)
        self.assertEqual(tokenizer.vocab[256], b"aa")
        self.assertIn((97, 97), tokenizer.merges)
        self.assertEqual(tokenizer.merges[(97, 97)], 256)

    def test_encode_simple(self):
        """Test encoding with a known vocabulary."""
        # Manually construct a tokenizer with known merges
        vocab = {x: bytes([x]) for x in range(256)}
        vocab[256] = b"ab"
        merges = {(97, 98): 256} # 'a', 'b' -> 'ab'
        
        tokenizer = BPEtokenizerParams(vocab, merges)
        
        text = "abc" # 'ab' + 'c' -> 256, 99
        encoded = tokenizer.encode(text)
        
        self.assertEqual(encoded, [256, 99])

    def test_decode_simple(self):
        """Test decoding back to string."""
        vocab = {x: bytes([x]) for x in range(256)}
        vocab[256] = b"ab"
        merges = {(97, 98): 256}
        
        tokenizer = BPEtokenizerParams(vocab, merges)
        
        ids = [256, 99] # 'ab', 'c'
        decoded = tokenizer.decode(ids)
        
        self.assertEqual(decoded, "abc")

    def test_roundtrip(self):
        """Test that encode -> decode gives back original text."""
        text = "The quick brown fox jumps over the lazy dog."
        # Train a small tokenizer
        tokenizer = train_bpe(text, num_merges=10)
        
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        self.assertEqual(decoded, text)

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        text = "hello"
        ids = [1, 2, 3, 4, 5] # 5 tokens
        # 'hello' is 5 bytes
        # ratio = 5 / 5 = 1.0
        
        ratio = BPEtokenizerParams.calculate_compression_ratio(text, ids)
        self.assertAlmostEqual(ratio, 1.0)
        
        ids_compressed = [1] # 1 token
        # ratio = 5 / 1 = 5.0
        ratio_compressed = BPEtokenizerParams.calculate_compression_ratio(text, ids_compressed)
        self.assertAlmostEqual(ratio_compressed, 5.0)

    def test_empty_string(self):
        """Test behavior with empty string."""
        # The current implementation raises ValueError on empty string because max() arg is empty
        # We assert this behavior since we cannot change the source code.
        with self.assertRaises(ValueError):
            train_bpe("", num_merges=1)
        
        # We can still test encode/decode on an empty string with a manually created tokenizer
        vocab = {x: bytes([x]) for x in range(256)}
        merges = {}
        tokenizer = BPEtokenizerParams(vocab, merges)
        
        encoded = tokenizer.encode("")
        self.assertEqual(encoded, [])
        
        decoded = tokenizer.decode([])
        self.assertEqual(decoded, "")
        
        ratio = tokenizer.calculate_compression_ratio("", [])
        self.assertEqual(ratio, 0)

if __name__ == '__main__':
    unittest.main()
