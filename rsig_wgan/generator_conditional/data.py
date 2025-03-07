"""
Implements data loading environment
"""

import torch
import math
import numpy as np
import yfinance as yf
import pandas as pd
from utils import *


class Data:
    def __init__(self, n_lags):
        self.n_lags = n_lags

    def generate(self, *kwargs):
        pass


class BM(Data):
    def __init__(self, n_lags, drift=0.0, std=1.0, dim=1, T=1.0):
        super().__init__(n_lags)
        self.drift = drift
        self.std = std
        self.dim = dim
        self.h = T / n_lags
        self.scaler = IDScaler()

    def generate(self, samples):
        path = torch.zeros([samples, self.n_lags, self.dim])
        path[:, 1:, :] = self.drift * self.h + math.sqrt(self.h) * self.std * torch.randn(samples, self.n_lags - 1,
                                                                                          self.dim)
        return torch.cumsum(path, 1)


class GBM(Data):
    def __init__(self, n_lags, drift=0.0, std=1.0, init=1.0, dim=1, T=1.0):
        super().__init__(n_lags)
        self.drift = drift
        self.std = std
        self.init = init
        self.dim = dim
        self.h = T / n_lags
        self.scaler = IDScaler()

    def generate(self, samples):
        path = torch.zeros([samples, self.n_lags, self.dim])
        path[:, 1:, :] = ((self.drift - self.std ** 2 / 2.) * self.h + math.sqrt(self.h) * self.std *
                          torch.randn(samples, self.n_lags - 1, self.dim))
        paths = self.init * torch.exp(torch.cumsum(path, 1))
        return paths


class AR(Data):
    def __init__(self, n_lags, phi, std=1.0, dim=1):
        super().__init__(n_lags)
        self.phi = phi
        self.std = std
        self.dim = dim
        self.scaler = Scaler()

    def generate(self, samples):
        paths = torch.zeros([samples, self.n_lags, self.dim])
        for i in range(1, self.n_lags):
            paths[:, i, :] = self.phi * paths[:, i - 1, :] + self.std * torch.randn(samples, self.dim)
        paths = self.scaler.transform(paths)
        return paths


class SP500(Data):
    def __init__(self, n_lags, start="2005-01-01", end="2023-10-31"):
        super().__init__(n_lags)
        self.start = start
        self.end = end
        self.scaler = Scaler()

    def generate(self):
        data = yf.download("SPY", start=self.start, end=self.end)
        log_returns = (np.log(data["Close"]) - np.log(data["Close"].shift(1)))[1:].to_numpy().reshape(-1, 1)
        log_returns = torch.from_numpy(log_returns).float().unsqueeze(0)
        log_returns = self.scaler.transform(log_returns)
        paths = rolling_window(log_returns, self.n_lags)
        return paths


class FOREX(Data):
    def __init__(self, n_lags):
        super().__init__(n_lags)
        self.scaler = Scaler()

    def generate(self):
        data = pd.read_csv("/content/drive/MyDrive/ColabNotebooks/Code_RSig_Gen/Unconditional_Generator/EURUSD1.csv", sep='\t')
        data.columns = ["Date", "Open", "High", "Low", "Close", "Vol"]
        log_returns = (np.log(data.Close) - np.log(data.Close).shift(1))[1:].to_numpy().reshape(-1, 1)
        log_returns = torch.from_numpy(log_returns).float().unsqueeze(0)
        log_returns = self.scaler.transform(log_returns)
        paths = rolling_window(log_returns, self.n_lags)
        return paths


class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None
        self.shift_by = None

    def transform(self, x):
        self.mean = x.mean()
        self.std = x.std()
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


class IDScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        self.shift_by = None

    def transform(self, x):
        return x

    def inverse(self, x):
        return x


def get_data(id):
    data, paths = None, None
    if id == "BM":
        data = BM(N_LAGS, DRIFT_BM, STD_BM, DIM)
        paths = data.generate(SAMPLES_BM)
    elif id == "GBM":
        data = GBM(N_LAGS, DRIFT_GBM, STD_GBM, INIT_GBM, DIM)
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
