import torch
import numpy as np
from torch.utils.data import Dataset

class KilterDataset(Dataset):
  def __init__(self, file_path):
    data = np.load(file_path)

    self.routes = torch.tensor(data['x'], dtype=torch.long)

    raw_coords = torch.tensor(data['xy'], dtype=torch.float32)

    # Adjust to go from -1 to 1
    self.mean_coords = raw_coords.mean(dim=0)
    self.std_coords = raw_coords.std(dim=0)

    # Safe division
    self.std_coords[self.std_coords == 0] = 1.0

    self.coords = (raw_coords - self.mean_coords) / self.std_coords

  def __len__(self):
    return len(self.routes)

  def __getitem__(self, idx):
    return self.routes[idx], self.coords