import torch.nn as nn
from GPT_config import GPT_CONFIG_124M

class FeedForward(nn.Module):
  def __init__(self, config:GPT_CONFIG_124M):
    super().__init__()
    
    self.layers = nn.Sequential(
      nn.Linear(config.emb_dim, config.emb_dim * 4),
      nn.GELU(),
      nn.Linear(config.emb_dim * 4, config.emb_dim)
    )
  
  def forward(self, x):
    return self.layers(x)