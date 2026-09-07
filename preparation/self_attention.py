import torch
import torch.nn as nn 

class MultiHeadAttention(nn.Module):
  def __init__(self, in_dim, out_dim, context_length, dropouts, num_heads, qkv_bias=False):
    super().__init__()
    
    # check if hte out dimention is devideable on num_heads
    assert (out_dim % num_heads == 0)
    
    self.out_dim = out_dim
    self.num_heads = num_heads
    self.heads_dim = out_dim // num_heads
    
    self.w_query = nn.Linear(in_dim, out_dim, bias=qkv_bias)
    self.w_key = nn.Linear(in_dim, out_dim, bias=qkv_bias)
    self.w_value = nn.Linear(in_dim, out_dim, bias=qkv_bias)
    
    self.register_buffer(
      "mask",
      torch.triu(
        torch.ones(context_length, context_length),
        diagonal=1
      )
    )
    
    self.dropout = nn.Dropout(dropouts)
    self.out_proj = nn.Linear(out_dim, out_dim)
    
  def forward(self, x:torch.tensor):
    
    b, num_tokens, in_dimention = x.shape
    
    keys = self.w_key(x)
    queries = self.w_query(x)
    values = self.w_value(x)
    
    # split the representation to create multi heads
    keys = keys.view(b, num_tokens, self.num_heads, self.heads_dim)
    values=values.view(b, num_tokens, self.num_heads, self.heads_dim)
    queries=queries.view(b, num_tokens, self.num_heads, self.heads_dim)
    
    # now we have this dimention for each representation: [b, num_token, num_head, head_dim]
    # but we need to tread the head dimention like the beatch dimention 
    # so we need to transpose them like this:
    keys = keys.transpose(1,2)
    queries = queries.transpose(1,2)
    values = values.transpose(1,2)
    
    attention_scores = queries @ keys.transpose(2,3)
    
    attention_scores.masked_fill_(
      self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
    )
    
    attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=1)
    
    attention_weights = self.dropout(attention_weights)
    
    context_vector = attention_weights @ values

    # move the dimention back:
    context_vector.transpose(1,2)
    # now we need to combine the last two dimention 
    # we need to use contgous function because we have already used transpose
    # and now we want to use some sort of reshaping on the object 
    context_vector = context_vector.contiguous().view(b, num_tokens, self.out_dim)
    # mix information across the concatenated heads
    context_vector = self.out_proj(context_vector)
    return context_vector
  
  
if __name__ == "__main__":
  # Hyperparameters
  in_dim = 768
  out_dim = 768
  context_length = 256
  dropouts = 0.1
  num_heads = 12

  # Create the attention layer
  attention = MultiHeadAttention(
      in_dim=in_dim,
      out_dim=out_dim,
      context_length=context_length,
      dropouts=dropouts,
      num_heads=num_heads
  )

  # Create fake Transformer input
  x = torch.randn(
      4,       # batch size
      10,      # number of tokens
      in_dim   # embedding dimension
  )

  # Pass through attention
  output = attention(x)

  print("Input shape: ", x.shape)
  print("Output shape:", output.shape)