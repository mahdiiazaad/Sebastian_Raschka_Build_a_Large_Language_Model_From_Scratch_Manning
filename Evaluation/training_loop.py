import torch.nn as nn
import torch
import tiktoken
from utils.encode_decode import token_ids_to_text, text_to_token_ids
from .text_evaluation import calc_loss_batch
from .text_evaluation import evaluate_model
from .text_generation import generate_text_simple

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model_simple(
  model:nn.Module,train_loader, val_loader, optimizer:torch.optim, device,
  num_epoch, eval_freq, eval_iter, start_context, tokenizer):
  
  # Keeping track of training progress
  train_loses, val_losses, track_tokens_seen = [], [], []
  
  tokens_seen, global_step = 0, -1
  
  for epoch in range(num_epoch):
    for inpu_batch, target_batch in train_loader:
      optimizer.zero_grad()
      
      loss = calc_loss_batch(input_batch=inpu_batch,
                             target_batch=target_batch,
                             model=model,
                             device=device)
      loss.backward()
      optimizer.step()
      
      tokens_seen += inpu_batch.numel()
      global_step += 1
      
      if global_step % eval_freq == 0 :
        train_loss, val_loss = evaluate_model(model=model,
                                               train_loader=train_loader,
                                               val_laoder=val_loader,
                                               device=device,
                                               eval_iter=eval_iter)
        train_loses.append(train_loss)
        val_losses.append(val_loss)
        track_tokens_seen.append(tokens_seen)
        
        print(f"Ep {epoch+1} (Step {global_step:06d}): "
              f"Train loss {train_loss:.3f}, "
              f"Val loss {val_loss:.3f}")
        generate_and_print_sample(
        model,
        tokenizer,
        device,
        start_context
    )
  
  return train_loses, val_losses, tokens_seen

  

def generate_and_print_sample(model:nn.Module,
                              tokenizer,
                              device,
                              start_context):
  model.eval()
  context_size = model.embeding_layer.position_embeding.weight.shape[0]
  
  encoded = text_to_token_ids(start_context, tokenizer=tokenizer).to(device)
  
  token_ids = generate_text_simple(model=model,
                                   in_idx=encoded,
                                   max_new_token=50,
                                   context_size=context_size)
  
  decoded_text = token_ids_to_text(token_ids=token_ids, tokenizer=tokenizer)
  print(decoded_text.replace("\n", " "))
  model.train()
  
if __name__ == "__main__":
  from pathlib import Path
  from GPT_model.GPT_2_model import GPTModel_2
  from GPT_model.GPT_config import GPT_CONFIG_124M
  from preparation.data_ingestion import create_data_loader
  
  text_path = Path(r"data\the-verdict\the-verdict.txt")
  with open(text_path) as f:
    raw_text = f.read()
    
    split_index = int(0.9 * len(raw_text))
    train_text = raw_text[:split_index]
    validation_index = raw_text[split_index:]
    
    train_dataloader = create_data_loader(train_text, shuffle=True, drop_last=True)
    val_dataloader = create_data_loader(validation_index, shuffle=False, drop_last=False)
  
  torch.manual_seed(42)
  model = GPTModel_2(GPT_CONFIG_124M)
  model.to(device=device)
  optimizer = torch.optim.AdamW(model.parameters(),
                                lr=0.0004,
                                weight_decay=0.1)
  
  tokenizer = tiktoken.get_encoding('gpt2')
  
  train_loses, val_losses, tokens_seen = train_model_simple(
    model=model,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    optimizer=optimizer,
    num_epoch=10,
    eval_iter=5,
    eval_freq=5,
    start_context="Every effort moves you",
    tokenizer=tokenizer,
    device=device
  )
  
  
  