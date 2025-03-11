import torch
import torch.nn as nn
import omegaconf
from typing import Union
from rsig_wgan.config import load_config
from .base import GeneratorBase


class NeuralSDEGenerator(GeneratorBase):
    def __init__(
        self,
        config: omegaconf.dictconfig.DictConfig,
        activation,
        device: str,
        A1: Union[torch.tensor, None],
        A2: Union[torch.tensor, None],
        xi1: Union[torch.tensor, None],
        xi2: Union[torch.tensor, None]
    ):
        super().__init__(config.neural_sde.input_dim, config.timeseries.data_dim)
        self.reservoir_dim = config.neural_sde.reservoir_dim_gen
        self.brownian_dim = config.neural_sde.brownian_dim
        self.activation = activation
        self.hidden_dim = config.neural_sde.hidden_dim
        self.device = device

        """
        Linear layers for initial condition NN
        """
        self.init_layer1 = nn.Linear(
           self.input_dim,
           self.hidden_dim,
           device=self.device
        )
        self.init_layer2 = nn.Linear(
           self.hidden_dim,
           self.reservoir_dim,
           device=self.device
        )

        """
        Sample random matrices and biases for reservoir 
        """
        self.rho1 = nn.Parameter(torch.randn(1, 1).to(self.device))
        self.rho2 = nn.Parameter(torch.randn(1, 1).to(self.device))
        self.rho3 = nn.Parameter(torch.randn(1, 1).to(self.device))
        self.rho4 = nn.Parameter(torch.randn(1, 1).to(self.device))

        if config.others.trainable_var:
          self.rho5 = nn.Parameter(torch.randn(1, 1).to(self.device))
        else: 
          self.rho5 = nn.Parameter(torch.ones(1, 1), requires_grad=False)

        if config.others.same_matrices:
          # Dimension of generator and metric needs to be the same
          assert self.reservoir_dim == config.rsigw1.reservoir_dim_metric

          self.B1, self.B2 = A1, A2
          self.lambda1, self.lambda2 = xi1, xi2
        else:
          self.B1, self.B2 = (torch.randn(self.reservoir_dim, self.reservoir_dim, device=self.device),
                        torch.randn(self.brownian_dim, self.reservoir_dim, self.reservoir_dim, device=self.device))

          self.lambda1, self.lambda2 = (torch.randn(self.reservoir_dim, 1, device=self.device),
                             torch.randn(self.brownian_dim, self.reservoir_dim, 1, device=self.device))
        
        self.activation = activation

        """
        Linear readout layer for the reservoir 
        """

        if config.others.time_homogeneous_readout:
          self.readouts = [nn.Linear(self.reservoir_dim, self.output_dim, device=self.device)] * config.timeseries.n_lags
        else:
          self.readouts = nn.ModuleList([nn.Linear(self.reservoir_dim, self.output_dim, device=self.device) for i in range(config.timeseries.n_lags)])

    def solve_neural_sde(self, V: torch.tensor, W: torch.tensor) -> torch.tensor:
        R = torch.empty(W.shape[0], W.shape[1], self.B1.shape[0], 1, device=self.device).clone()
        R[:, 0, :] = V.clone()

        for t in range(1, W.shape[1]):
            R[:, t, :] = (R[:, t - 1, :].clone() + self.activation(self.rho1 * self.B1 @ R[:, t - 1, :].clone() + self.rho2 * self.lambda1)+ torch.sum(self.activation(self.rho3 * self.B2 @ R[:, t - 1, :].unsqueeze(-3).clone()
                                                      + self.rho4 * self.lambda2)
                                      @ self.rho5 * (W[:, t, :, None, None] - W[:, t - 1, :, None, None]), axis=1))
                         
        return R

    def forward(self, batch_size: int, n_lags: int) -> torch.tensor:
        """
        :param batch_size: number of samples
        :param n_lags: number of time steps
        :param device: depends on system setup (cpu or gpu)
        :return: tensor of synthetic data
        """

        V = torch.randn(batch_size, self.input_dim, device=self.device)
        V = self.init_layer1(V)
        V = self.activation(V)
        V = self.init_layer2(V)
        V = torch.reshape(V, (batch_size, self.reservoir_dim, 1))
        increments = torch.randn(batch_size, n_lags, self.brownian_dim, device=self.device)
        W = torch.cumsum(increments, 1)
        W[:, 0, :] = 0.0

        R = self.solve_neural_sde(V, W)

        for n in range(n_lags):
            if n == 0:
                x = self.readouts[n](R[:, n].reshape(R[:, n].shape[0], -1))
            else:
                x = torch.cat((x, self.readouts[n](R[:, n].reshape(R[:, n].shape[0], -1))), 1)

        return x.reshape(x.shape[0], x.shape[1], 1)