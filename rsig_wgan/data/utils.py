import numpy as np
import omegaconf
import torch

from .data import FOREX, SP500, AutoregressiveProcess, BrownianMotion


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

def chronological_split(
    x: torch.tensor, n_lags: int, ratio_train=0.7, ratio_val=0.1
) -> torch.tensor:
    """
    Chronological split for overlapping rolling windows.

    Windows i and j share a return whenever |i - j| <= n_lags - 1, so a random split
    leaks almost every test window into training. Dropping n_lags - 1 windows between
    the blocks is exactly enough to make them share no return.
    """
    size = x.shape[0]
    embargo = n_lags - 1

    train_end = int(size * ratio_train)
    val_start = train_end + embargo
    val_end = val_start + int(size * ratio_val)
    test_start = val_end + embargo

    if test_start >= size:
        raise ValueError(
            f"{size} windows are too few for a {ratio_train}/{ratio_val} split with an "
            f"embargo of {embargo} windows"
        )

    return x[:train_end], x[val_start:val_end], x[test_start:]


def to_numpy(x: torch.tensor) -> np.array:
    return x.detach().cpu().numpy()

def get_data(
    config: omegaconf.dictconfig.DictConfig
) -> torch.tensor:
    data, paths = None, None
    if config.data.id == "BM":
        data = BrownianMotion(
                config.timeseries.n_lags,
                config.bm.drift,
                config.bm.std,
                config.timeseries.data_dim
            )
        paths = data.generate(config.bm.samples)
    elif config.data.id == "SP500":
        data = SP500(config.timeseries.n_lags)
        paths = data.generate()
    elif config.data.id == "AR":
        data = AutoregressiveProcess(config.timeseries.n_lags, config.ar.phi)
        paths = data.generate(config.ar.samples)
    elif config.data.id == "FOREX":
        data = FOREX(config.timeseries.n_lags)
        paths = data.generate()
    if config.data.id in ("SP500", "FOREX"):
        return [data, chronological_split(paths, config.timeseries.n_lags)]
    return [data, train_test_split(paths)]