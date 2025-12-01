import torch 
from torch.utils.data import Dataset, DataLoader 
from BPE_from_scratch import train_bpe, download_tinystories, load_and_preprocess
from torch.nn import Embedding

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self,idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader(txt, tokenizer, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader 

def token_embedding(vocab_size, output_dim):
    torch.manual_seed(42)
    embedding_layer = Embedding(vocab_size, output_dim)
    return embedding_layer

def pos_encoding(max_length, output_dim):
    pos_enc_layer = torch.nn.Embedding(max_length, output_dim)
    pos_embeddings = pos_enc_layer(torch.arange(max_length))
    return pos_embeddings

if __name__ == "__main__":
    filename = "TinyStories-valid.txt"
    download_tinystories(filename)
    full_text = load_and_preprocess(filename)

    # --- SANITY CHECK 1: Is the text file actually English? ---
    print(f"Text Length: {len(full_text)}")
    print(f"First 100 chars: {repr(full_text[:100])}")
    # If this prints empty strings or weird symbols, delete the .txt file and re-download.

    print("Training Tokenizer...")
    tokenizer = train_bpe(full_text[:10000], num_merges=100) 

    # --- SANITY CHECK 2: What is ID 65? ---
    # If your previous output was all 65s, let's see what 65 means.
    print(f"Token 65 decodes to: '{tokenizer.decode([65])}'")

    print("Creating DataLoader...")
    dataloader = create_dataloader(
        txt=full_text, 
        tokenizer=tokenizer,
        batch_size=2,
        max_length=4,
        stride=4
    )

    print("\n--- Visual Inspection ---")
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
    inputs, targets = next(data_iter)

    print(f"Input Shape: {inputs.shape}")
    print(f"Target Shape: {targets.shape}")
    print(f"Input example: {inputs[0]}")
    print(f"Target example: {targets[0]}")
    
    is_shifted = inputs[0][1] == targets[0][0]
    print(f"Shift logic holds: {is_shifted}")

    print("\n--- 4. TEST EMBEDDING COMPATIBILITY ---")
   
    NUM_MERGES = 100
    VOCAB_SIZE = 256 + NUM_MERGES
    embedding_dim = 16
    emb_layer = token_embedding(VOCAB_SIZE, embedding_dim)
    token_embeddings = emb_layer(inputs)
    try:
        print(f"Passing inputs to Embedding Layer (Vocab={VOCAB_SIZE}, Dim={embedding_dim})...")
        embedded_x = emb_layer(inputs)
        print(f"Output Embedding Shape: {embedded_x.shape}")
        print(emb_layer.weight)
        print("✅ SUCCESS: Indices fit within Embedding layer.")
        print(emb_layer(torch.tensor([3])))
    except IndexError as e:
        print(f"❌ FAILURE: Index Error. Your tokenizer produced an ID larger than the Embedding vocab size.")
        print(f"Max ID in input: {inputs.max()}, Embedding Size: {VOCAB_SIZE}")
        print(f"Error details: {e}")

    print("--TESTING POS ENCODING")
    pos_embeddings = pos_encoding(max_length=4, output_dim=embedding_dim)
    print(pos_embeddings)
    print(pos_embeddings.shape)
    print("✅ SUCCESS: Pos Embedding shape matches expected.")
    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings)
    print(input_embeddings.shape)
    print("✅ SUCCESS: Input Embedding shape matches expected.")
