"""
This file contains the setup of the implemented generator models:
    - Conditional Neural SDE Reservoir
"""

import torch
import torch.nn as nn
from rsigcw1 import *


class GeneratorBase(nn.Module):
    def __init__(self, noise_input_dim: int, output_dim: int):
        super(GeneratorBase, self).__init__()
        self.noise_input_dim = noise_input_dim
        self.output_dim = output_dim

    def forward(self, batch_size: int, n_lags: int, device: str = DEVICE):
        """
        to be specified for the individual generator
        """
        pass


class ConditionalNeuralSDEGenerator(GeneratorBase):
    def __init__(self, input_dim: int, output_dim: int, reservoir_dim: int, brownian_dim: int, activation,
                 hidden_dim: int = 32, initial_noise_dim: int = 30, device = DEVICE):
        super().__init__(input_dim, output_dim)
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.brownian_dim = brownian_dim
        self.activation = activation
        self.device = device

        """
        Linear layers for initial condition NNs
        """
        self.hidden_dim = hidden_dim
        self.initial_noise_dim = initial_noise_dim
        self.hidden_layer_init_1 = nn.Linear(self.input_dim, self.hidden_dim, device = self.device)
        self.output_layer_init_1 = nn.Linear(self.hidden_dim, self.initial_noise_dim, device = self.device)

        self.hidden_layer_init_2 = nn.Linear(self.reservoir_dim, self.hidden_dim, device = self.device)
        self.output_layer_init_2 = nn.Linear(self.hidden_dim, self.reservoir_dim - self.initial_noise_dim, device = self.device)

        """
        Sample random matrices and biases for conditional reservoir 
        """

        self.rho1, self.rho2, self.rho3, self.rho4 = (nn.Parameter(torch.randn(1, 1)).to(self.device), nn.Parameter(torch.randn(1, 1)).to(self.device),
                                                      nn.Parameter(torch.randn(1, 1)).to(self.device), nn.Parameter(torch.randn(1, 1)).to(self.device))

        self.B1, self.B2 = A1, A2

        self.lambda1, self.lambda2 = xi1, xi2

        """
        Sample random matrices and biases for conditioning reservoir of path past  
        """

        self.A1, self.A2 = A1, A2

        self.xi1, self.xi2 = xi1, xi2

        self.activation = activation

        """
        Linear readout layer for the reservoir 
        """

        self.readout = nn.Linear(self.reservoir_dim, self.output_dim, device = DEVICE)

    def solve_conditional_neural_sde(self, rsig_cond: torch.tensor, V: torch.tensor, W: torch.tensor) -> torch.tensor:
        R = torch.empty(W.shape[0], W.shape[1], self.B1.shape[0], 1, device = self.device)
        R[:, 0, :] = torch.cat((rsig_cond, V), axis=1)

        for t in range(1, W.shape[1]):
            R[:, t, :] = (R[:, t - 1, :].clone()
                          + self.activation(self.rho1 * self.B1 @ R[:, t - 1, :].clone() + self.rho2 * self.lambda1)
                          + self.activation(self.rho3 * self.B2 @ R[:, t - 1, :].clone()
                                            + self.rho4 * self.lambda2) @ (W[:, t, :, None].clone() - W[:, t - 1, :, None].clone()))

        return R

    def forward(self, batch_size: int, n_lags: int, x_past: torch.tensor, device: str = DEVICE) -> torch.tensor:
        V = torch.randn(batch_size, self.noise_input_dim, device = device)
        V = self.hidden_layer_init_1(V)
        V = self.activation(V)
        V = self.output_layer_init_1(V)
        V = torch.reshape(V, (batch_size, self.initial_noise_dim, 1))

        rsig_cond = (compute_rsig(x_past, self.A1, self.A2, self.xi1, self.xi2, self.reservoir_dim, self.activation).
                     reshape(1, -1)).to(device)
        rsig_cond = self.hidden_layer_init_2(rsig_cond)
        rsig_cond = self.activation(rsig_cond)
        rsig_cond = self.output_layer_init_2(rsig_cond).reshape(1, self.reservoir_dim - self.initial_noise_dim, 1)
        rsig_cond = rsig_cond.repeat(batch_size, 1, 1).requires_grad_()

        increments = torch.randn(batch_size, n_lags, self.brownian_dim, device = device)
        W = torch.cumsum(increments, 1)
        W[:, 0, :] = 0.0

        R = self.solve_conditional_neural_sde(rsig_cond, V, W)

        for n in range(n_lags):
            if n == 0:
                x = self.readout(R[:, n].reshape(R[:, n].shape[0], -1))
            else:
                x = torch.cat((x, self.readout(R[:, n].reshape(R[:, n].shape[0], -1))), 1)

        return x.reshape(x.shape[0], x.shape[1], 1)
