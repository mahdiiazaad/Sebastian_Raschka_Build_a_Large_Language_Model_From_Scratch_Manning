import torch
import tiktoken

def text_to_token_ids(
    start_context: str,
    tokenizer: tiktoken.Encoding
):
    encoded = tokenizer.encode(start_context)
    return torch.tensor(encoded).unsqueeze(0)
  
def token_ids_to_text(token_ids, tokenizer: tiktoken.Encoding):
    token_ids = token_ids.squeeze(0)
    return tokenizer.decode(token_ids.tolist())