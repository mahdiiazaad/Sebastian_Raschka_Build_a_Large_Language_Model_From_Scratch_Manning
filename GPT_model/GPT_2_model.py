import torch.nn as nn

from .GPT_config import GPT_CONFIG_124M
from .layer_normalization import LayerNorm
from .transformer_block import TransformerBlock
from preparation.embeding import TransformerInput

# config = GPT_CONFIG_124M()
class GPTModel_2(nn.Module):
  def __init__(self, config:GPT_CONFIG_124M):
    super().__init__()
    
    self.embeding_layer = TransformerInput(config.vocab_size, 
                                           config.context_length,
                                           config.emb_dim)
    self.transformers = nn.Sequential(
      *[TransformerBlock(config=config) 
      for _ in range(config.n_layers)]
      )
    
    self.norm_layer = LayerNorm(config.emb_dim)
    
    self.dropout = nn.Dropout(config.drop_rate)
    
    self.output_head = nn.Linear(config.emb_dim,
                                config.vocab_size,
                                bias=False)
    
  def forward(self, in_idx):
    
    # seperating the dimentions
    batch_size, seq_len = in_idx.shape
    
    # token and position embeding
    x = self.embeding_layer(in_idx)
    # drop out layer 
    x = self.dropout(x)
    # transformers (12) layer
    x = self.transformers(x)
    # final layer norm
    x = self.norm_layer(x)
    
    logits = self.output_head(x)
    return logits
    
     
    
    
    
  