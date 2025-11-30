import torch 
from torch.utils.data import Dataset, DataLoader 
from BPE_from_scratch import train_bpe, download_tinystories, load_and_preprocess

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

if __name__ == "__main__":
    filename = "TinyStories-valid.txt"
    download_tinystories(filename)
    full_text = load_and_preprocess(filename)

    # 2. Prepare Tokenizer 
    # Since we aren't loading a saved one, we train it quickly on a subset
    print("Training Tokenizer...")
    # Using the function imported from bpe.py
    tokenizer = train_bpe(full_text[:10000], num_merges=100) 

    # 3. Create Data Loader
    print("Creating DataLoader...")
    dataloader = create_dataloader(
        txt=full_text[:5000], # Use a small slice for testing
        tokenizer=tokenizer,
        batch_size=2,
        max_length=4,
        stride=4
    )

    # 4. Verify Output
    print("\n--- Visual Inspection ---")
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print(f"Input Shape: {inputs.shape}")
    print(f"Target Shape: {targets.shape}")
    print(f"Input example: {inputs[0]}")
    print(f"Target example: {targets[0]}")