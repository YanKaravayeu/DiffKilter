import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class KilterTransformer(nn.Module):
  def __init__(self, num_classes, hidden_dim, num_layers, nhead):
    super().__init__()

    self.token_emb = nn.Embedding(num_classes, hidden_dim)

    self.pos_emb = SpatialCoordEmbedding(hidden_dim)

    self.time_mlp = nn.Sequential(
        SinusoidalPositionEmbeddings(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim * 2),
        nn.GELU(),
        nn.Linear(hidden_dim * 2, hidden_dim)
    )

    layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
    self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    self.to_logits = nn.Linear(hidden_dim, num_classes)

  def forward(self, x, t, coords):
      x_emb = self.token_emb(x)

      pos_emb = self.pos_emb(coords)

      t_emb = self.time_mlp(t)

      h = x_emb + pos_emb + t_emb.unsqueeze(1)

      h = self.transformer(h)

      return self.to_logits(h)
  

class SpatialCoordEmbedding(nn.Module):
  def __init__(self, hidden_dim):
    super().__init__()
    self.projection = nn.Sequential(
        nn.Linear(2, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim)
    )

  def forward(self, coords):
    return self.projection(coords)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # time shape: (Batch_Size,)
        device = time.device
        half_dim = self.dim // 2

        # Calculate the frequencies for the sine/cosine waves
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]

        # Interleave sine and cosine waves
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        # Handle odd dimensions just in case
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))

        return embeddings
    
