import torch 
from typing import Callable
from sklearn.linear_model import LinearRegression

def compute_rsig(
    path: torch.tensor,
    A1: torch.tensor,
    A2: torch.tensor,
    xi1: torch.tensor,
    xi2: torch.tensor,
    res_dim: int,
    activation: Callable,
    device: str
) -> torch.tensor:
    rsig = torch.zeros([path.shape[0], res_dim, 1]).to(device)

    for i in range(path.shape[1]):
        rsig = (rsig + activation(A1 @ rsig + xi1)
                + torch.sum(activation(A2 @ rsig.unsqueeze(-3) + xi2) @ path[:, i, :, None, None].to(device), axis=1))

    return rsig


def lr_rsig(x_future: torch.tensor, x_past: torch.tensor, A1: torch.tensor, A2: torch.tensor, xi1: torch.tensor,
            xi2: torch.tensor, dim: int, activation, terminal=True):
    reservoir_future = compute_rsig(x_future, A1, A2, xi1, xi2, dim, activation, self.device).reshape([x_future.shape[0], dim])
    reservoir_past = compute_rsig(x_past, A1, A2, xi1, xi2, dim, activation, self.device).reshape([x_past.shape[0], dim])
    X, y = to_numpy(reservoir_past), to_numpy(reservoir_future)
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X, y)
    return torch.from_numpy(lr.predict(X)).float()