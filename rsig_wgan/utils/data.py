import numpy as np
import torch

def to_numpy(x: torch.tensor) -> np.array:
    return x.detach().cpu().numpy()

def sample_indices(dataset_size, batch_size: int) -> torch.tensor:
    indices = torch.from_numpy(np.random.choice(dataset_size, size=batch_size, replace=False))
    return indices.long()