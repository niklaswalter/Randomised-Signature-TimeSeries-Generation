import torch
import torch.nn as nn
from rsig_wgan.config import load_config
from base import GeneratorBase


class NeuralSDEGenerator(GeneratorBase):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        reservoir_dim: int,
        brownian_dim: int,
        activation,
        hidden_dim: int,
        device: str,
        A1: torch.tensor | None,
        A2: torch.tensor | None,
        xi1: torch.tensor | None,
        xi2: torch.tensor | None
    ):
        super().__init__(input_dim, output_dim)
        self.reservoir_dim = reservoir_dim
        self.brownian_dim = brownian_dim
        self.activation = activation
        self.device = device
        self.A1, self.A2 = A1, A2
        self.xi1, self.xi2 = xi1, xi2

        """
        Linear layers for initial condition NN
        """
        self.hidden_dim = hidden_dim
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

        if load_config()['others']['trainable_var']:
          self.rho5 = nn.Parameter(torch.randn(1, 1).to(self.device))
        else: 
          self.rho5 = nn.Parameter(torch.ones(1, 1), requires_grad=False)

        if load_config()['others']['same_matrices']:
          # Dimension of generator and metric needs to be the same
          assert load_config()['neural_sde']['reservoir_dim_gen'] == load_config()['rsigw1']['reservoir_dim_metric']

          self.B1, self.B2 = B1, B2
          self.lambda1, self.lambda2 = lambda1, lambda2
        else:
          self.B1, self.B2 = (torch.randn(RESERVOIR_DIM_GEN, RESERVOIR_DIM_GEN, device = DEVICE),
                        torch.randn(BROWNIAN_DIM, RESERVOIR_DIM_GEN, RESERVOIR_DIM_GEN, device = DEVICE))

          self.lambda1, self.lambda2 = (torch.randn(RESERVOIR_DIM_GEN, 1, device = DEVICE),
                             torch.randn(BROWNIAN_DIM, RESERVOIR_DIM_GEN, 1, device = DEVICE))
        
        self.activation = activation

        """
        Linear readout layer for the reservoir 
        """

        if TIME_HOMOGENEOUS_READOUT:
          self.readouts = [nn.Linear(self.reservoir_dim, self.output_dim, device = DEVICE)] * N_LAGS
        else:
          self.readouts = nn.ModuleList([nn.Linear(self.reservoir_dim, self.output_dim, device=DEVICE) for i in range(N_LAGS)])

    def solve_neural_sde(self, V: torch.tensor, W: torch.tensor) -> torch.tensor:
        R = torch.empty(W.shape[0], W.shape[1], self.B1.shape[0], 1, device=DEVICE).clone()
        R[:, 0, :] = V.clone()

        for t in range(1, W.shape[1]):
            R[:, t, :] = (R[:, t - 1, :].clone() + self.activation(self.rho1 * self.B1 @ R[:, t - 1, :].clone() + self.rho2 * self.lambda1)+ torch.sum(self.activation(self.rho3 * self.B2 @ R[:, t - 1, :].unsqueeze(-3).clone()
                                                      + self.rho4 * self.lambda2)
                                      @ self.rho5 * (W[:, t, :, None, None] - W[:, t - 1, :, None, None]), axis=1))
                         
        return R

    def forward(self, batch_size: int, n_lags: int, device: str=DEVICE) -> torch.tensor:
        """
        :param batch_size: number of samples
        :param n_lags: number of time steps
        :param device: depends on system setup (cpu or gpu)
        :return: tensor of synthetic data
        """

        V = torch.randn(batch_size, self.input_dim, device=device)
        V = self.init_layer1(V)
        V = self.activation(V)
        V = self.init_layer2(V)
        V = torch.reshape(V, (batch_size, self.reservoir_dim, 1))
        increments = torch.randn(batch_size, n_lags, self.brownian_dim, device=device)
        W = torch.cumsum(increments, 1)
        W[:, 0, :] = 0.0

        R = self.solve_neural_sde(V, W)

        for n in range(n_lags):
            if n == 0:
                x = self.readouts[n](R[:, n].reshape(R[:, n].shape[0], -1))
            else:
                x = torch.cat((x, self.readouts[n](R[:, n].reshape(R[:, n].shape[0], -1))), 1)

        return x.reshape(x.shape[0], x.shape[1], 1)