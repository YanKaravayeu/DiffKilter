import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class KilterTransformer(nn.Module):
  """
  Custom Transformer Encoder build for Discrete Absorbing Diffusion.

  Model processes climbing routes spatially. Combines token embeddings,
  spatial embeddings, and time embeddings.

  These embeddings are summed together and passed through a standard PyTorch
  Transformer Encoder to predict the original, unmasked climbing route.

  Args:
    num_classes (int): Total number of token classes (6 for this project).
    hidden_dim (int): Dimensionality of the model's internal representations.
    num_layers (int): Number of Transformer Encoder blocks.
    nhead (int): Number of attention heads in the Multi-Head Attention mechanism.
  """

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
      """
      Forward pass of the KilterTransformer.

      Args:
        x (torch.Tensor): The masked board state, shape (Batch, 476).
        t (torch.Tension): The current diffusion timesteps, shape (Batch,).
        coords (torch.Tensor): The normalised spatial coordinates, shape (Batch, 476, 2).

        Returns:
            torch.Tensor: The unnormalised logit predictions for the unmasked board,
                          shape (Batch, 476, num_classes).
      """

      x_emb = self.token_emb(x)

      pos_emb = self.pos_emb(coords)

      t_emb = self.time_mlp(t)

      h = x_emb + pos_emb + t_emb.unsqueeze(1)

      h = self.transformer(h)

      return self.to_logits(h)
  

class SpatialCoordEmbedding(nn.Module):
  """
  Embeds physical 2D spatial coordinates into a high-dimensional latent space.

  Allows the Transformer to understand physical distances between holds on the Kilterboard,
  rather than just treating it all as a sequence.

  Args:
        hidden_dim (int): The dimensionality of the output embedding.
  """

  def __init__(self, hidden_dim):

    super().__init__()
    self.projection = nn.Sequential(
        nn.Linear(2, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim)
    )

  def forward(self, coords):
    """
    Args:
        coords (torch.Tensor): Tensor of spatial coordinates, shape (Batch, 476, 2).

    Returns:
        torch.Tensor: Embedded coordinates, shape (Batch, 476, hidden_dim).
    """
    
    return self.projection(coords)


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Encodes the current diffusion timestep 't' in a high-dimensional vector.

    Diffusion models need to know how much noise has been added to the board.
    Instead of passing a single integer, this class converts t into a continuous
    wave of sin and cos frequencies, allowing Transformer to process t better.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time (torch.Tensor): A 1D tensor of timesteps, shape (Batch_size).

            Returns:
                torch.Tensor: The time embeddings, shape (Batch_size, dim).
        """

        device = time.device
        half_dim = self.dim // 2

        #  Calculate the frequencies for the sine/cosine waves
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]

        #  Interleave sine and cosine waves
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        #  Handle odd dimensions just in case
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))

        return embeddings
    
