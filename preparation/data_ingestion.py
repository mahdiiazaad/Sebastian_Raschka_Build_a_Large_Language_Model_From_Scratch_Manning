import os
import torch
import tiktoken
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
  """
Initialize the GPT dataset by reading, tokenizing, and splitting text
    into overlapping input-target sequences.
  """
  def __init__(self, text_path: Path, tokenizer: tiktoken.Encoding, max_length, stride):
    super().__init__()
    
    self.input_ids, self.target_ids = [], []
    
    with open(text_path, 'r', encoding='utf-8') as f:
      raw_text = f.read()
      
        
    token_ids = tokenizer.encode(raw_text)
    
    for i in range(0, len(token_ids) - max_length, stride):
      
      input_chunk = token_ids[i : i + max_length]
      target_chunk = token_ids[i + 1: i + max_length+1]
      
      self.input_ids.append(input_chunk)
      self.target_ids.append(target_chunk)
      
  def __len__(self):
    return len(self.input_ids)
  
  def __getitem__(self, index):
    return torch.tensor(self.input_ids[index]), torch.tensor(self.target_ids[index])
  
def create_data_loader(
  text_path: Path,
  batch_size=4,
  max_length=256,
  stride=128,
  shuffle=True,
  drop_last=True,
  num_workers=os.cpu_count()
    ):
  
  tokenizer = tiktoken.get_encoding('gpt2')
  
  data_set = GPTDatasetV1(text_path=text_path,
                          tokenizer=tokenizer,
                          max_length=max_length,
                          stride=stride)
  
  dataloader = DataLoader(dataset=data_set,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          drop_last=drop_last,
                          shuffle=shuffle)
  
  return dataloader


if __name__ == "__main__":
  text_path = Path(r"data\the-verdict\the-verdict.txt")
  dataloader = create_data_loader(text_path)
  
  for x, y in dataloader:
    print(x.shape, y.shape)
    break


  
  
  
  
    