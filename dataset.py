import torch
import numpy as np
from torch.utils.data import Dataset

class KilterDataset(Dataset):
  """
  Loads and preprocesses Kilterboard climbing routes from a compress NumPy array.

  Loads categorical route data, but also extracts and normalises physical 2D spatial
  coordinates of the board.
  
  Args:
    file_path (str): The path to the '.npz' dataset file.
  """

  def __init__(self, file_path):
    data = np.load(file_path)

    self.routes = torch.tensor(data['x'], dtype=torch.long)

    raw_coords = torch.tensor(data['xy'], dtype=torch.float32)

    #  Adjust to go from -1 to 1
    self.mean_coords = raw_coords.mean(dim=0)
    self.std_coords = raw_coords.std(dim=0)

    #  Safe division
    self.std_coords[self.std_coords == 0] = 1.0

    self.coords = (raw_coords - self.mean_coords) / self.std_coords

  def __len__(self):
    """
    Returns:
      int: The total number of climbing routes in the dataset.
    """
    return len(self.routes)

  def __getitem__(self, idx):
    """
    Retrieves a single climbing route and its corresponding spatial coordinates.

    Args:
      idx (int): The index of the route to retrieve.

    Returns:
      tuple:
        torch.Tensor: The route array of class integers, shape (476,).
        torch.Tensor: The normalised (x, y) coordinates, shape (476, 2).
    """
    return self.routes[idx], self.coords