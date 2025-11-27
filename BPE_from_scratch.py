import re
from collections import defaultdict
import os
import requests

def download_tinystories(filename="TinyStories-valid.txt"):
   
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
    
    if os.path.exists(filename):
        print(f"{filename} already exists.")
        return

    print(f"Downloading {filename}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download: {e}")


class BPEtokenizerParams: 
    def __init__(self, vocab, merges): 
        self.vocab = vocab 
        self.merges = merges 
        

    def encode(self, string: str) -> list[int]: 
        ids = list(string.encode("utf-8"))
        work_string = "".join(chr(b) for b in ids)
        for (p0, p1), new_index in self.merges.items(): 
            pair_str = chr(p0) + chr(p1)
            token_str = chr(new_index)

            work_string = work_string.replace(pair_str, token_str)
        return [ord(c) for c in work_string]
    
    def decode(self, indices: list[int]) -> str: 
 
        bytes_list = []
        for idx in indices:
            if idx in self.vocab:
                bytes_list.append(self.vocab[idx])
            else:
                pass 
                
        string = b"".join(bytes_list).decode("utf-8", errors="replace")
        return string
    
    @staticmethod
    def calculate_compression_ratio(text, encoded_ids):
        num_bytes = len(text.encode("utf-8"))
        num_tokens = len(encoded_ids)
        if num_tokens == 0: return 0
        return num_bytes / num_tokens

def load_and_preprocess(dataset_path):
    
    with open(dataset_path, 'r', encoding='utf-8') as f: 
        raw_text = f.read()
    return raw_text


def train_bpe(string, num_merges):
    indices = "".join(chr(b) for b in string.encode("utf-8"))
    merges: dict[tuple[int, int], int] = {}
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    
    for i in range(num_merges): 
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):
            counts[ord(index1), ord(index2)] += 1
        
        pair = max(counts, key=counts.get)
        index1, index2 = pair 

        new_index = 256+i
        merges[pair] = new_index
        vocab[new_index] = vocab[index1] + vocab[index2]
        
        pair_str = chr(index1) + chr(index2)
        new_token_str = chr(new_index)
        indices = indices.replace(pair_str, new_token_str)
   
    return BPEtokenizerParams(vocab = vocab, merges = merges)


if __name__ == "__main__":
    filename = "TinyStories-valid.txt"
    download_tinystories(filename)

    full_text = load_and_preprocess(filename)
    training_data = full_text[:1_000_000]

    tokenizer = train_bpe(training_data, num_merges=3)
    validation_sample = full_text[1_000_000:1_005_000] 

    print("\n--- Visual Inspection ---")
    print(f"Vocab Size: {len(tokenizer.vocab)}")
    sample = "I am just testing a small BPE tokenizer :)."
    ids = tokenizer.encode(sample)
    print(f"'{sample}' -> {ids}")
    print(f"Back -> '{tokenizer.decode(ids)}'")
    val_ids = tokenizer.encode(validation_sample)
    ratio = tokenizer.calculate_compression_ratio(validation_sample, val_ids)
    print(f"Compression Ratio on Validation Set: {ratio:.2f}X")