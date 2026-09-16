"""
Implements the LSTM generator
"""

import torch
import torch.nn as nn
import omegaconf
from typing import Tuple

from .base import GeneratorBase


class ResidualBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.create_residual_connection = True if input_dim == output_dim else False

    def forward(self, x: torch.tensor) -> torch.tensor:
        y = self.linear(x)
        y = self.activation(y)
        if self.create_residual_connection:
            y = x + y
        return y


class ResFNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int], flatten: bool = False):
        super(ResFNN, self).__init__()
        blocks = list()
        self.input_dim = input_dim
        self.flatten = flatten
        input_dim_block = input_dim
        for hidden_dim in hidden_dims:
            blocks.append(ResidualBlock(input_dim_block, hidden_dim))
            input_dim_block = hidden_dim
        blocks.append(nn.Linear(input_dim_block, output_dim))
        self.network = nn.Sequential(*blocks)
        self.blocks = blocks

    def forward(self, x: torch.tensor) -> torch.tensor:
        if self.flatten:
            x = x.reshape(x.shape[0], -1)
        return self.network(x)


def init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class LSTMGenerator(GeneratorBase):
    def __init__(
        self,
        config: omegaconf.dictconfig.DictConfig,
        device: str
    ):
        super().__init__(config.lstm.input_dim, config.timeseries.data_dim)
        self.hidden_dim = config.lstm.hidden_dim
        self.num_layers = config.lstm.num_layers
        self.device = device

        self.rnn = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(self.hidden_dim, self.output_dim, bias=True)
        self.linear.apply(init_weights)

        self.initial_nn = nn.Sequential(
            ResFNN(self.input_dim, self.hidden_dim * self.num_layers, [self.hidden_dim, self.hidden_dim]),
            nn.Tanh()
        )
        self.initial_nn.apply(init_weights)

        self.to(self.device)

    def forward(self, batch_size: int, n_lags: int) -> torch.tensor:
        """
        :param batch_size: number of samples
        :param n_lags: number of time steps
        :return: tensor of synthetic data
        """

        z = (0.1 * torch.randn(batch_size, n_lags, self.input_dim, device=self.device))
        z[:, 0, :] *= 0
        z = z.cumsum(1)

        z0 = torch.randn(batch_size, self.input_dim, device=self.device)
        h0 = self.initial_nn(z0).view(
            batch_size, self.rnn.num_layers, self.rnn.hidden_size
        ).permute(1, 0, 2).contiguous()

        c0 = torch.zeros_like(h0)
        h1, _ = self.rnn(z, (h0, c0))
        x = self.linear(h1)

        assert x.shape[1] == n_lags

        return x
