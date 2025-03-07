import torch

def rolling_window(x: torch.tensor, n_lags: int) -> torch.tensor:
    return torch.cat([x[:, t:t + n_lags] for t in range(x.shape[1] - n_lags + 1)], dim=0)