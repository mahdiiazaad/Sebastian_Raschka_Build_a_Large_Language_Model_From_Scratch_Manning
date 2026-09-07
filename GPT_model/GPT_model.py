import torch.nn as nn
from GPT_config import GPT_CONFIG_124M
from preparation.embeding import TransformerInput
from preparation.self_attention import MultiHeadAttention

class GPTModelV1(nn.Module):
  def __init__(self, config:GPT_CONFIG_124M, embeding_layer:nn.Module ):
    super().__init__()
    
    self.embeding_layer = TransformerInput(config.vocab_size, 
                                      config.context_length, 
                                      config.emb_dim)
    
