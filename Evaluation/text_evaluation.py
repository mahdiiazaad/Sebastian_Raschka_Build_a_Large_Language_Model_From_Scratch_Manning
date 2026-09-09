import torch
import torch.nn as nn
from torch.utils.data import dataloader

from pathlib import Path
from GPT_model.GPT_config import GPT_CONFIG_124M
from GPT_model.GPT_2_model import GPTModel_2
from sklearn.model_selection import train_test_split
from preparation.data_ingestion import create_data_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def calc_loss_batch(input_batch, target_batch, model, device):
  input_batch, target_batch = input_batch.to(device), target_batch.to(device)
  
  logits = model(input_batch)
  loss = nn.functional. cross_entropy(
    logits.flatten(0,1),
    target_batch.flatten()
  )
  
  return loss


def calc_loss_loader(model: nn.Module, dataloader:dataloader, 
                     device=device, num_batches=None):
  # Initialize the loss
  total_loss = 0.
  
  # check if the dataloader is empty
  # prevent devision by 0 errir
  if len(dataloader) == 0:
    return float('nan')
  
  # Decide how many batches to evaluate 
  if num_batches is None:
    num_batches = len(dataloader)
  else:
    num_batches = min(
      num_batches,
      len(dataloader)
    )
  
  for i, (input_batch, target_batch) in enumerate(dataloader):
    if i < num_batches:
      loss = calc_loss_batch(
        input_batch=input_batch,
        target_batch=target_batch,
        model=model,
        device=device
      )
      
      total_loss += loss.item()
    else:
      break
    
  return total_loss / num_batches

def evaluate_model(model:nn.Module,
                   train_loader,
                   val_laoder,
                   device,
                   eval_iter):
  model.eval()
  with torch.no_grad():
    train_loss = calc_loss_loader(
      model=model,
      dataloader=train_loader,
      device=device,
      num_batches=eval_iter
    )
    
    val_loss = calc_loss_loader(
      model=model,
      dataloader=val_laoder,
      device=device,
      num_batches=eval_iter
    )

    model.train()
    
  return train_loss, val_loss


if __name__ == "__main__":
  text_path = Path(r"data\the-verdict\the-verdict.txt")
  config = GPT_CONFIG_124M()
  with open(text_path) as f:
    raw_text = f.read()
    
    split_index = int(0.9 * len(raw_text))
    train_text = raw_text[:split_index]
    validation_text = raw_text[split_index:]    
    train_dataloader = create_data_loader(train_text, 
                                          shuffle=True, drop_last=True)
    test_dataloader = create_data_loader(validation_text, 
                                         shuffle=False, drop_last=False)
    
    model = GPTModel_2(config=config)
    model.to(device=device)
    
    with torch.no_grad():
      train_loss = calc_loss_loader(dataloader=train_dataloader, model=model, device=device)
      
      val_loss = calc_loss_loader(dataloader=test_dataloader, model=model, device=device)
  
  print(train_loss)
  print(val_loss)