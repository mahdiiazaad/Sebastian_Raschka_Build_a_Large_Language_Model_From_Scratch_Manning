import torch.nn as nn


from GPT_config import GPT_CONFIG_124M
from layer_normalization import LayerNorm
from feed_forward_network import FeedForward
from preparation.self_attention import MultiHeadAttention
# what we need:
# 1. multihead attention class
# 2. feed forward 
# 3. layer norm 1 & 2
# 4. drop out + the original

class TransformerBlock(nn.Module):
  def __init__(self, config: GPT_CONFIG_124M):
    super().__init__()
    
    self.attention = MultiHeadAttention(
      in_dim=config.emb_dim,
      out_dim=config.emb_dim,
      context_length=config.context_length,
      dropouts=config.drop_rate,
      qkv_bias=config.qkv_bias,
      num_heads=config.n_heads
    )
    self.feedforward = FeedForward(config=config)
    
    self.layernorm_1 = LayerNorm(config.emb_dim)
    self.layernorm_2 = LayerNorm(config.emb_dim)
    
    self.dropout = nn.Dropout(p=config.drop_rate)
    
  def forward(self, x):
    # create a nstance of the x to be added to the dropout 
    short_cut = x
    
    # first norm layer 
    x = self.layernorm_1(x)
    
    # multi head attention 
    X = self.attention(x)
    
    # drop out 
    x = self.dropout(x) + short_cut
    
    # second norm layer
    x = self.layernorm_2(x)
    
    # feed forward 
    x = self.feedforward(x)
    
    # second dropout 
    x = self.dropout(x) + short_cut
    
    
    return x