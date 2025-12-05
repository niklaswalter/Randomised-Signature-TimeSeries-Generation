"""
Implements the RSig-Wasserstein-1 metric and the corresponding
training procedure of the generator
"""

from collections import defaultdict

import torch
from torch import optim
from copy import deepcopy
from tqdm import tqdm
from typing import Union
from rsig_wgan.config import activation_mapping

from rsig_wgan.discriminator_models.utils import l2_dist

"""
Computes the randomised signature for an input path
"""

def compute_rsig(
    path: torch.tensor,
    A1: torch.tensor,
    A2: torch.tensor,
    xi1: torch.tensor,
    xi2: torch.tensor,
    res_dim: int,
    activation_mapping: str,
    device: str
) -> torch.tensor:
    rsig = torch.zeros([path.shape[0], res_dim, 1]).to(device)

    activation = activation_mapping[activation]

    for i in range(path.shape[1]):
        rsig = (rsig + activation(A1 @ rsig + xi1)
                + torch.sum(activation(A2 @ rsig.unsqueeze(-3) + xi2) @ path[:, i, :, None, None].to(device), axis=1))

    return rsig


"""
Computes the terminal difference of the randomised signature for an input path
"""

def compute_rsig_td(
    path: torch.tensor,
    A1: torch.tensor,
    A2: torch.tensor,
    xi1: torch.tensor,
    xi2: torch.tensor,
    res_dim: int,
    activation: str,
    device: str
) -> torch.tensor:
    rsig = torch.zeros([path.shape[0], res_dim, 1]).to(device)

    activation = activation_mapping[activation]

    for i in range(path.shape[1]):
        if i == path.shape[1] - 1:
            rsig_penulti = rsig.clone()
        rsig = (rsig + activation(A1 @ rsig + xi1)
                + torch.sum(activation(A2 @ rsig.unsqueeze(-3) + xi2) @ path[:, i, :, None, None].to(device), axis=1))

    return rsig - rsig_penulti


class RSigW1Metric:
    """
    Class for implementation of RSig-W1 metric
    """

    def __init__(
        self,
        x_real: torch.tensor,
        config,
        A1: torch.tensor,
        A2: torch.tensor,
        xi1: torch.tensor,
        xi2: torch.tensor,
        device: str,
        terminal_diff: bool = True
    ):
        self.x_real = x_real
        self.res_dim = config.rsigw1.reservoir_dim_metric
        self.activation = config.neural_sde.activation
        self.A1 = A1
        self.A2 = A2
        self.xi1 = xi1
        self.xi2 = xi2
        self.device = device
        self.terminal_diff = terminal_diff
        self.name = "RSig-W1-Dist"

        if self.terminal_diff:
            self.expected_rsig_real = compute_rsig_td(self.x_real, self.A1, self.A2, self.xi1, self.xi2, self.res_dim,
                                                      self.activation, self.device).mean(0).to(self.device)
        else:
            self.expected_rsig_real = compute_rsig(self.x_real, self.A1, self.A2, self.xi1, self.xi2, self.res_dim,
                                                   self.activation).mean(0).to(self.device)

    def __call__(self, x_fake: torch.tensor) -> float:
        if self.terminal_diff:
            expected_rsig_fake = compute_rsig_td(x_fake, self.A1, self.A2, self.xi1, self.xi2, self.res_dim,
                                                 self.activation, self.device).mean(0).to(self.device)
        else:
            expected_rsig_fake = compute_rsig(x_fake, self.A1, self.A2, self.xi1, self.xi2, self.res_dim,
                                              self.activation).mean(0).to(self.device)

        return l2_dist(self.expected_rsig_real, expected_rsig_fake)


class RSigWGANTraining:
    """
    Class for training procedure with RSig-W1 discriminator
    """

    def __init__(
        self,
        x_train: torch.tensor,
        x_val: torch.tensor,
        generator,
        config,
        device: str,
        A1: Union[torch.tensor, None],
        A2: Union[torch.tensor, None],
        xi1: Union[torch.tensor, None],
        xi2: Union[torch.tensor, None]
    ):
        self.x_train = x_train
        self.x_val = x_val
        self.batch_size = config.hyperparameters.batch_size
        self.n_lags = self.x_train.shape[1]
        self.generator = generator
        self.generator_optim = optim.Adam(self.generator.parameters())
        self.best_generator = None
        self.num_grad_steps = config.hyperparameters.gradient_steps
        self.learning_rate = config.hyperparameters.learning_rate
        self.res_dim = config.rsigw1.reservoir_dim_metric
        self.data_dim = config.timeseries.data_dim
        self.activation = config.neural_sde.activation
        self.device = device

        self.A1, self.A2 = A1, A2
        self.xi1, self.xi2 = xi1, xi2

        self.train_losses_history = defaultdict(list)
        self.val_losses_history = defaultdict(list)

        self.metric = RSigW1Metric(
                            x_real=self.x_train,    
                            config=config,
                            A1=self.A1,
                            A2=self.A2, 
                            xi1=self.xi1, 
                            xi2=self.xi2, 
                            terminal_diff=True, 
                            device=device
                        )
        self.metric_val = RSigW1Metric(
                            x_real=self.x_val,
                            config=config,
                            A1=self.A1,
                            A2=self.A2, 
                            xi1=self.xi1, 
                            xi2=self.xi2, 
                            terminal_diff=True, 
                            device=device
                        )
        self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.generator_optim, gamma=0.95, step_size=128)

    """
    Method to fit model using Adam optimiser
    """

    def fit(self):
        self.generator.to(self.device)
        best_loss = None

        for j in tqdm(range(self.num_grad_steps)):
            self.generator_optim.zero_grad()
            x_fake = self.generator(batch_size=self.batch_size, n_lags=self.n_lags)
            loss = self.metric(x_fake)
            loss.backward()
            best_loss = loss.item() if j == 0 else best_loss
            if (j + 1) % 100 == 0:
                val_loss = self.metric_val(x_fake)
                self.val_losses_history["RSigW1Val"].append(val_loss.item())
                print("rsig-w1 - train loss: {:1.2e}, best train loss: {:1.2e}, val loss: {:1.2e}"
                      .format(loss.item(), best_loss, val_loss))
            self.generator_optim.step()
            self.scheduler.step()
            self.train_losses_history["RSigW1Loss"].append(loss.item())
            if loss < best_loss:
                self.best_generator = deepcopy(self.generator.state_dict())
                best_loss = loss
        self.generator.load_state_dict(self.best_generator)
