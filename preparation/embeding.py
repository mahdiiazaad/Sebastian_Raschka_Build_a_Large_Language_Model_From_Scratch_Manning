import torch
import torch.nn as nn

class TransformerInput(nn.Module):
  def __init__(self, vocab_size, context_length, embedding_dim):
    super().__init__()
    
    self.token_embeding = nn.Embedding(vocab_size, embedding_dim=embedding_dim)
    self.position_embeding = nn.Embedding(context_length, embedding_dim=embedding_dim)
    
    
  def forward(self, input_ids):
    
    batch_size, sequence_length = input_ids.shape
    
    token_embeddings = self.token_embeding(input_ids)
    
    positions = torch.arange(sequence_length, device=input_ids.device)
    
    position_embedings = self.position_embeding(positions)
    
    x = token_embeddings + position_embedings
    return x
    
if __name__ == '__main__':
  model = TransformerInput(
    vocab_size=1000,
    context_length=10,
    embedding_dim=8
)

  # Fake input
  input_ids = torch.tensor([
      [10, 25, 73, 42],
      [5,  91, 12, 88]
  ])

  # Pass input through the model
  output = model(input_ids)

  print("Input shape:", input_ids.shape)
  print("Output shape:", output.shape)
  print("Output:")
  print(output)