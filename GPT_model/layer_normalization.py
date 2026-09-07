import torch
import torch.nn as nn

class LayerNorm(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()
    
    self.eps = 1e-5
    self.shift = nn.Parameter(torch.zeros(embed_dim))
    self.scale = nn.Parameter(torch.ones(embed_dim))
    
  def forward(self, x):
    
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, unbiased=False)
    
    normed_x = (x - mean) / (torch.sqrt(var + self.eps))
    
    return self.scale * normed_x + self.shift