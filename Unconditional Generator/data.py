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
from utils import *


class Data:
    """
    The parent class for data classes
    """

    def __init__(self, n_lags: int):
        self.n_lags = n_lags

    def generate(self, *kwargs):
        pass


class BM(Data):
    """
    Class implementing generation of Brownian motion paths
    """

    def __init__(self, n_lags: int, drift: double = 0.0, std: double = 1.0, dim: int = 1, T: double = 1.0):
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


class AR(Data):
    """
    Class implementing generation of paths of AR(1) process
    """

    def __init__(self, n_lags: int, phi: double, std: double = 1.0, dim: int = 1):
        super().__init__(n_lags)
        self.phi = phi
        self.std = std
        self.dim = dim
        self.scaler = Scaler()

    def generate(self, samples: int) -> torch.tensor:
        paths = torch.zeros([samples, self.n_lags, self.dim])
        for i in range(1, self.n_lags):
            paths[:, i, :] = self.phi * paths[:, i - 1, :] + self.std * torch.randn(samples, self.dim)
        paths = self.scaler.transform(paths)
        return paths


class SP500(Data):
    """
    Class loading S&P 500 log-returns and applying a rolling window
    """

    def __init__(self, n_lags: int, start: str = "2005-01-01", end: str = "2023-10-31"):
        super().__init__(n_lags)
        self.start = start
        self.end = end
        self.scaler = Scaler()

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
        self.scaler = Scaler()

    def generate(self) -> torch.tensor:
        data = pd.read_csv("/content/drive/MyDrive/ColabNotebooks/Code_RSig_Gen/Unconditional_Generator/EURUSD1.csv",
                           sep='\t')
        data.columns = ["Date", "Open", "High", "Low", "Close", "Vol"]
        log_returns = (np.log(data.Close) - np.log(data.Close).shift(1))[1:].to_numpy().reshape(-1, 1)
        log_returns = torch.from_numpy(log_returns).float().unsqueeze(0)
        log_returns = self.scaler.transform(log_returns)
        paths = rolling_window(log_returns, self.n_lags)
        return paths


class Scaler:
    """
    Class for the standardisation of data
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.shift_by = None

    def transform(self, x: torch.tensor) -> torch.tensor:
        self.mean = x.mean()
        self.std = x.std()
        return (x - self.mean) / self.std

    def inverse(self, x: torch.tensor) -> torch.tensor:
        return x * self.std + self.mean


class IDScaler:
    """
    Class for scaler applying the identity function
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.shift_by = None

    def transform(self, x: torch.tensor) -> torch.tensor:
        return x

    def inverse(self, x: torch.tensor) -> torch.tensor:
        return x


"""
Function returning the sample paths for given data ID
"""

def get_data(id: str) -> torch.tensor:
    data, paths = None, None
    if id == "BM":
        data = BM(N_LAGS, DRIFT_BM, STD_BM, DATA_DIM)
        paths = data.generate(SAMPLES_BM)
    elif id == "GBM":
        data = GBM(N_LAGS, DRIFT_GBM, STD_GBM, INIT_GBM, DATA_DIM)
        paths = data.generate(SAMPLES_BM)
    elif id == "SP500":
        data = SP500(N_LAGS)
        paths = data.generate()
    elif id == "AR":
        data = AR(N_LAGS, PHI)
        paths = data.generate(SAMPLES_AR)
    elif id == "FOREX":
        data = FOREX(N_LAGS)
        paths = data.generate()
    return [data, train_test_split(paths)]
