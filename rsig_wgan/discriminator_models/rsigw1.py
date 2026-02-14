"""
Implements the RSig-Wasserstein-1 metric and the corresponding
training procedure of the generator
"""

from collections import defaultdict
from copy import deepcopy
from typing import Union

import torch
from loguru import logger
from torch import optim
from tqdm import tqdm

from rsig_wgan.config import ACTIVATION_REGISTRY
from rsig_wgan.discriminator_models.utils import l2_dist
from rsig_wgan.utils import compute_rsig


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
    ):
        self.x_real = x_real
        self.res_dim = config.rsigw1.reservoir_dim_metric
        self.activation = ACTIVATION_REGISTRY[config.neural_sde.activation]
        self.A1 = A1
        self.A2 = A2
        self.xi1 = xi1
        self.xi2 = xi2
        self.device = device
        self.name = "RSig-W1-Dist"

        self.expected_rsig_real = compute_rsig(
            self.x_real,
            self.A1,
            self.A2,
            self.xi1,
            self.xi2,
            self.res_dim,
            self.activation,
            self.device
        ).mean(0).to(self.device)

    def __call__(self, x_fake: torch.tensor) -> float:
        expected_rsig_fake = compute_rsig(
            x_fake,
            self.A1,
            self.A2,
            self.xi1,
            self.xi2,
            self.res_dim,
            self.activation,
            self.device
        ).mean(0).to(self.device)

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
        self.scheduler = optim.lr_scheduler.StepLR(
            optimizer=self.generator_optim,
            gamma=0.95,
            step_size=128
        )

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
                logger.info("rsig-w1 - train loss: {:1.2e}, best train loss: {:1.2e}, val loss: {:1.2e}"
                      .format(loss.item(), best_loss, val_loss))
            self.generator_optim.step()
            self.scheduler.step()
            self.train_losses_history["RSigW1Loss"].append(loss.item())
            if loss < best_loss:
                self.best_generator = deepcopy(self.generator.state_dict())
                best_loss = loss
        self.generator.load_state_dict(self.best_generator)
