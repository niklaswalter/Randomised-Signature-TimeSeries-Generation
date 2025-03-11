"""
Implements data loading environment for following data types:
    - Brownian motion
    - AR(1) process
    - S&P500 log-returns
    - FOREX EUR/USD log-returns
"""

import torch
import math
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from .scaling import Standardiser, IDScaler


class Data:
    """
    The parent class for data classes
    """

    def __init__(self, n_lags: int):
        self.n_lags = n_lags

    def generate(self, *kwargs):
        pass


class BrownianMotion(Data):
    """
    Class implementing generation of Brownian motion paths
    """

    def __init__(self, n_lags: int, drift: float = 0.0, std: float = 1.0, dim: int = 1, T: float = 1.0):
        super().__init__(n_lags)
        self.drift = drift
        self.std = std
        self.dim = dim
        self.h = T / n_lags
        self.scaler = IDScaler()

    def generate(self, samples: int) -> torch.tensor:
        path = torch.zeros([samples, self.n_lags, self.dim])
        path[:, 1:, :] = self.drift * self.h + math.sqrt(self.h) * self.std * torch.randn(samples, self.n_lags - 1,
                                                                                          self.dim)
        return torch.cumsum(path, 1)


class AutoregressiveProcess(Data):
    """
    Class implementing generation of paths of AR(1) process
    """

    def __init__(self, n_lags: int, phi: float, std: float = 1.0, dim: int = 1):
        super().__init__(n_lags)
        self.phi = phi
        self.std = std
        self.dim = dim
        self.scaler = Standardiser()

    def generate(self, samples: int) -> torch.tensor:
        paths = torch.zeros([samples, self.n_lags, self.dim])
        for i in range(1, self.n_lags):
            paths[:, i, :] = self.phi * paths[:, i - 1, :] + self.std * torch.randn(samples, self.dim)
        paths = self.scaler.transform(paths)
        return paths


def rolling_window(x: torch.tensor, n_lags: int) -> torch.tensor:
    return torch.cat([x[:, t:t + n_lags] for t in range(x.shape[1] - n_lags + 1)], dim=0)

class SP500(Data):
    """
    Class loading S&P 500 log-returns and applying a rolling window
    """

    def __init__(self, n_lags: int, start: str = "2005-01-01", end: str = "2023-10-31"):
        super().__init__(n_lags)
        self.start = start
        self.end = end
        self.scaler = Standardiser()

    def generate(self) -> torch.tensor:
        data = yf.download("SPY", start=self.start, end=self.end)
        log_returns = (np.log(data["Close"]) - np.log(data["Close"].shift(1)))[1:].to_numpy().reshape(-1, 1)
        log_returns = torch.from_numpy(log_returns).float().unsqueeze(0)
        log_returns = self.scaler.transform(log_returns)
        paths = rolling_window(log_returns, self.n_lags)
        return paths


class FOREX(Data):
    """
    Class loading FOREX EUR/USD log-returns and applying a rolling window
    """

    def __init__(self, n_lags: int):
        super().__init__(n_lags)
        self.scaler = Standardiser()
        self.file_path = Path(__file__).parent / "EURUSD1.csv"

    def generate(self) -> torch.tensor:
        data = pd.read_csv(self.file_path, sep='\t')
        data.columns = ["Date", "Open", "High", "Low", "Close", "Vol"]
        log_returns = (np.log(data.Close) - np.log(data.Close).shift(1))[1:].to_numpy().reshape(-1, 1)
        log_returns = torch.from_numpy(log_returns).float().unsqueeze(0)
        log_returns = self.scaler.transform(log_returns)
        paths = rolling_window(log_returns, self.n_lags)
        return paths
