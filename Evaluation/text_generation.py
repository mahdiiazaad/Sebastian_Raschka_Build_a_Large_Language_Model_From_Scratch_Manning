import torch.nn as nn
import torch

def generate_text_simple(model:nn.Module, in_idx, max_new_token, context_size):
  for _ in range(max_new_token):
    idx_con = in_idx[:, -context_size:]
    
    with torch.no_grad():
      logits = model(in_idx)
      # selecting the last generated token vocab scores
      logits = logits[:, -1, :] 
            
			# turn the vocab scores in to probability distribution
      probas = torch.softmax(logits, dim=-1)
            
			# select teh highest token with the higest probability
      idx_next = torch.argmax(
          probas,
          dim=-1,
          keepdim=True
      )
			
	    # add the new generated token to the idx list (currect token sequence)
      in_idx = torch.cat((in_idx, idx_next), dim=1)
      idx = in_idx

  return idx
      
       