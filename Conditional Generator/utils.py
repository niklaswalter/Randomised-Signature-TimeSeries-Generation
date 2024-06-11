"""
Contains helper functions
"""

import torch
import torch.nn as nn
import numpy as np
from config import *


def l2_dist(x, y: float) -> float:
    return (x - y).pow(2).sum().sqrt()


def to_numpy(x: torch.tensor) -> np.array:
    return x.detach().cpu().numpy()


def rolling_window(x: torch.tensor, n_lags: int) -> torch.tensor:
    return torch.cat([x[:, t:t + n_lags] for t in range(x.shape[1] - n_lags + 1)], dim=0)


def sample_indices(dataset_size, batch_size: int) -> torch.tensor:
    indices = torch.from_numpy(np.random.choice(dataset_size, size=batch_size, replace=False))
    return indices.long()


def train_test_split(x: torch.tensor, ratio_train=0.2, ratio_val=0.1) -> torch.tensor:
    size = x.shape[0]
    train_size = int(size * ratio_train)
    val_size = int(size * ratio_val)
    indices_train = sample_indices(size, train_size)
    indices_wo_train = torch.LongTensor([i for i in range(size) if i not in indices_train])
    indices_val = sample_indices(indices_wo_train, val_size)
    indices_test = torch.LongTensor([i for i in indices_wo_train if i not in indices_val])

    x_train = x[indices_train]
    x_val = x[indices_val]
    x_test = x[indices_test]
    return x_train, x_val, x_test


def get_activation(id):
    if id == "Sigmoid":
        return nn.Sigmoid()
    elif id == "Tanh":
        return nn.Tanh()


def cov(x, rowvar=False, bias=True, ddof=None, aweights=None):
    x = to_numpy(x)
    _, L, C = x.shape
    x = x.reshape(-1, L*C)
    return torch.from_numpy(np.cov(x, rowvar=False)).float()


def cov_diff(x_real, x_fake):
    cov_real, cov_fake = cov(x_real), cov(x_fake)
    return torch.norm(cov_real - cov_fake, p = 'fro')


def acf(x, lag, dim=(0, 1)):
    acf_list = list()
    x = x - x.mean((0, 1))
    std = torch.var(x, unbiased=False, dim=(0, 1))
    for i in range(lag):
        y = x[:, i:] * x[:, :-i] if i > 0 else torch.pow(x, 2)
        acf_i = torch.mean(y, dim) / std
        acf_list.append(acf_i)
    if dim == (0, 1):
        return torch.stack(acf_list)
    else:
        return torch.cat(acf_list, 1)


def acf_diff(x_real, x_fake, lag, dim=(0, 1)):
    return l2_dist(acf(x_real, lag), acf(x_fake, lag))
