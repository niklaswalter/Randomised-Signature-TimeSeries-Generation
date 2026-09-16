from typing import Callable

import torch
from sklearn.linear_model import LinearRegression

from .data import to_numpy


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


def reservoir_features(
    x: torch.tensor,
    A1: torch.tensor,
    A2: torch.tensor,
    xi1: torch.tensor,
    xi2: torch.tensor,
    dim: int,
    activation: Callable,
    device: str
):
    return to_numpy(compute_rsig(x, A1, A2, xi1, xi2, dim, activation, device).reshape([x.shape[0], dim]))


def fit_lr_rsig(
    x_future: torch.tensor,
    x_past: torch.tensor,
    A1: torch.tensor,
    A2: torch.tensor,
    xi1: torch.tensor,
    xi2: torch.tensor,
    dim: int,
    activation: Callable,
    device: str
) -> LinearRegression:
    """
    Fit the conditional expectation of the future randomised signature given the past one
    """
    X = reservoir_features(x_past, A1, A2, xi1, xi2, dim, activation, device)
    y = reservoir_features(x_future, A1, A2, xi1, xi2, dim, activation, device)
    estimator = LinearRegression(fit_intercept=True)
    estimator.fit(X, y)
    return estimator


def predict_lr_rsig(
    estimator: LinearRegression,
    x_past: torch.tensor,
    A1: torch.tensor,
    A2: torch.tensor,
    xi1: torch.tensor,
    xi2: torch.tensor,
    dim: int,
    activation: Callable,
    device: str
) -> torch.tensor:
    """
    Apply a fitted estimator to pasts it was not fitted on
    """
    X = reservoir_features(x_past, A1, A2, xi1, xi2, dim, activation, device)
    return torch.from_numpy(estimator.predict(X)).float()


def lr_rsig(
    x_future: torch.tensor,
    x_past: torch.tensor,
    A1: torch.tensor,
    A2: torch.tensor,
    xi1: torch.tensor,
    xi2: torch.tensor,
    dim: int,
    activation: Callable,
    device: str
) -> torch.tensor:
    estimator = fit_lr_rsig(x_future, x_past, A1, A2, xi1, xi2, dim, activation, device)
    return predict_lr_rsig(estimator, x_past, A1, A2, xi1, xi2, dim, activation, device)
