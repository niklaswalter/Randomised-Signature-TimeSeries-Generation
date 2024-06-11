"""
This file contains the setup of the implemented generator models:
    - Neural SDE
    - LSTM
"""

import torch
import torch.nn as nn
from typing import Tuple
from config import *

class GeneratorBase(nn.Module):
    """
    The parent class for generator models
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super(GeneratorBase, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, batch_size: int, n_lags: int, device: str = 'cpu'):
        """
        to be specified for the individual generator
        """
        pass


class NeuralSDEGenerator(GeneratorBase):
    """
    Class implementing the NeuralSDE generator model
    """
    
    def __init__(self, input_dim: int, output_dim: int, reservoir_dim: int, brownian_dim: int, activation,
                 hidden_dim: int = 32, device: str = DEVICE):
        super().__init__(input_dim, output_dim)
        self.reservoir_dim = reservoir_dim
        self.brownian_dim = brownian_dim
        self.activation = activation
        self.device = device

        """
        Linear layers for initial condition NN
        """
        self.hidden_dim = hidden_dim
        self.init_layer1 = nn.Linear(self.input_dim, self.hidden_dim, device = self.device)
        self.init_layer2 = nn.Linear(self.hidden_dim, self.reservoir_dim, device = self.device)

        """
        Sample random matrices and biases for reservoir 
        """
        self.rho1, self.rho2, self.rho3, self.rho4 = (nn.Parameter(torch.randn(1, 1).to(self.device)), nn.Parameter(torch.randn(1, 1).to(self.device)),
                                                      nn.Parameter(torch.randn(1, 1).to(self.device)), nn.Parameter(torch.randn(1, 1).to(self.device)))

        
        self.B1, self.B2 = (torch.randn(self.reservoir_dim, self.reservoir_dim, device = DEVICE),
                            torch.randn(self.brownian_dim, self.reservoir_dim, self.reservoir_dim, device = DEVICE))

        self.lambda1, self.lambda2 = (torch.randn(self.reservoir_dim, 1, device = DEVICE),
                                      torch.randn(self.brownian_dim, self.reservoir_dim, 1, device = DEVICE))

        self.activation = activation

        """
        Linear readout layers for the reservoir 
        """
        
        self.readouts = nn.ModuleList([nn.Linear(self.reservoir_dim, self.output_dim, device=DEVICE)
                                       for i in range(10)])

    def solve_neural_sde(self, V: torch.tensor, W: torch.tensor) -> torch.tensor:
        """
        Methods solving the NeuralSDE numerically using an EM scheme
        """
        R = torch.empty(W.shape[0], W.shape[1], self.B1.shape[0], 1, device=DEVICE).clone()
        R[:, 0, :] = V.clone()

        for t in range(1, W.shape[1]):
            R[:, t, :] = (R[:, t - 1, :].clone() + self.activation(self.rho1 * self.B1 @ R[:, t - 1, :].clone()
                          + self.rho2 * self.lambda1) + torch.sum(self.activation(self.rho3 * self.B2 @ R[:, t - 1, :].unsqueeze(-3).clone()
                          + self.rho4 * self.lambda2)
                          @ (W[:, t, :, None, None] - W[:, t - 1, :, None, None]), axis=1))
                         
        return R

    def forward(self, batch_size: int, n_lags: int, device: str = DEVICE) -> torch.tensor:
        """
        Methods implementing forward call
        """

        V = torch.randn(batch_size, self.input_dim, device = device)
        V = self.init_layer1(V)
        V = self.activation(V)
        V = self.init_layer2(V)
        V = torch.reshape(V, (batch_size, self.reservoir_dim, 1))
        increments = torch.randn(batch_size, n_lags, self.brownian_dim, device = device)
        W = torch.cumsum(increments, 1)
        W[:, 0, :] = 0.0

        R = self.solve_neural_sde(V, W)

        for n in range(n_lags):
            if n == 0:
                x = self.readouts[n](R[:, n].reshape(R[:, n].shape[0], -1))
            else:
                x = torch.cat((x, self.readouts[n](R[:, n].reshape(R[:, n].shape[0], -1))), 1)

        return x.reshape(x.shape[0], x.shape[1], 1)


class ResidualBlock(nn.Module):
    """
    Class for single residual block
    """

    def __init__(self, input_dim: int, output_dim: int):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.create_residual_connection = True if input_dim == output_dim else False

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Methods implementing forward call
        """
        y = self.linear(x)
        y = self.activation(y)
        if self.create_residual_connection:
            y = x + y
        return y


class ResFNN(nn.Module):
    """
    Class for ResFNN for initial NN in LSTM generator
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int], flatten: bool = False):
        super(ResFNN, self).__init__()
        blocks = list()
        self.input_dim = input_dim
        self.flatten = flatten
        input_dim_block = input_dim
        for hidden_dim in hidden_dims:
            blocks.append(ResidualBlock(input_dim_block, hidden_dim))
            input_dim_block = hidden_dim
        blocks.append(nn.Linear(input_dim_block, output_dim, device = DEVICE))
        self.network = nn.Sequential(*blocks)
        self.blocks = blocks

    def forward(self, x):
        if self.flatten:
            x = x.reshape(x.shape[0], -1)
        out = self.network(x)
        return out


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        try:
            nn.init.zeros_(m.bias)
        except:
            pass


class LSTMGenerator(GeneratorBase):
    """
    Class implementing LSTM generator
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, device: str = DEVICE):
        super().__init__(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                           device = device)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=True, device = device)
        self.linear.apply(init_weights)
        self.device = device

        self.initial_nn = nn.Sequential(ResFNN(input_dim, hidden_dim * num_layers, [hidden_dim, hidden_dim]),
                                        nn.Tanh())
        self.initial_nn.apply(init_weights)

    def forward(self, batch_size, n_lags, **kwargs) -> torch.tensor:
        """
        Method implementing forward call
        """

        z = (0.1 * torch.randn(batch_size, n_lags, self.input_dim)).to(self.device)
        z[:, 0, :] *= 0
        z = z.cumsum(1)

        z0 = torch.randn(batch_size, self.input_dim, device=self.device)
        h0 = self.initial_nn(z0).view(batch_size, self.rnn.num_layers, self.rnn.hidden_size).permute(1, 0,
                                                                                                     2).contiguous()

        c0 = torch.zeros_like(h0)
        h1, _ = self.rnn(z, (h0, c0))
        x = self.linear(h1)

        assert x.shape[1] == n_lags

        return x