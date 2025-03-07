"""
Implements the conditional RSig-Wasserstein-1 metric and the
corresponding training procedure of the generator
"""

import torch
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
from torch import optim
import sklearn
from sklearn.linear_model import LinearRegression
from utils import *


# -----------------------------------------------------------------------------------------
# Computation of the terminal difference of the randomised signature for an input path
# -----------------------------------------------------------------------------------------

def compute_rsig_terminal(path: torch.tensor, A1: torch.tensor, A2: torch.tensor, xi1: torch.tensor, xi2: torch.tensor,
                          dim: int, activation, device: str = DEVICE) -> torch.tensor:
    rsig = torch.zeros([path.shape[0], dim, 1]).to(device)

    for i in range(path.shape[1]):
        if i == path.shape[1] - 1:
            rsig_1 = rsig.clone()
        rsig = rsig + activation(A1 @ rsig + xi1) + activation(A2 @ rsig + xi2) @ path[:, i, :, None].to(device)

    return rsig - rsig_1


# -----------------------------------------------------------------------------------------
# Computation of the randomised signature for an input path
# -----------------------------------------------------------------------------------------

def compute_rsig(path: torch.tensor, A1: torch.tensor, A2: torch.tensor, xi1: torch.tensor, xi2: torch.tensor,
                 dim: int, activation) -> torch.tensor:
    rsig = torch.zeros([path.shape[0], dim, 1]).to(DEVICE)

    for i in range(path.shape[1]):
        rsig = rsig + activation(A1 @ rsig + xi1) + activation(A2 @ rsig + xi2) @ path[:, i, :, None].to(DEVICE)

    return rsig


# -----------------------------------------------------------
# Implements the linear regression approximation of the
# future reservoir from the past reservoir
# -----------------------------------------------------------

def lr_rsig(x_future: torch.tensor, x_past: torch.tensor, A1: torch.tensor, A2: torch.tensor, xi1: torch.tensor,
            xi2: torch.tensor, dim: int, activation, terminal=True):
    if terminal:
        reservoir_future = (compute_rsig_terminal(x_future, A1, A2, xi1, xi2, dim, activation).
                            reshape([x_future.shape[0], dim]))
        reservoir_past = (compute_rsig_terminal(x_past, A1, A2, xi1, xi2, dim, activation).
                          reshape([x_past.shape[0], dim]))
    else:
        reservoir_future = compute_rsig(x_future, A1, A2, xi1, xi2, dim, activation).reshape([x_future.shape[0], dim])
        reservoir_past = compute_rsig(x_past, A1, A2, xi1, xi2, dim, activation).reshape([x_past.shape[0], dim])
    X, y = to_numpy(reservoir_past), to_numpy(reservoir_future)
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X, y)
    return torch.from_numpy(lr.predict(X)).float()


# -----------------------------------------------------------
# Implements the Conditional RSig-Wasserstein1 metric
# -----------------------------------------------------------

class RSigCW1Metric:
    def __init__(self, x_real, p, q, indices, N, activation, A1, A2, xi1, xi2, device=DEVICE,
                 terminal=True):
        self.indices = indices
        self.N = N
        self.activation = activation
        self.device = device
        self.terminal = terminal

        self.A1, self.A2 = A1, A2
        self.xi1, self.xi2 = xi1, xi2

        self.name = "RSig-Cond-W1-Dist"

        self.x_real = x_real
        self.n_lags = self.x_real.shape[1]
        self.p = p
        self.q = q
        self.x_real_past = x_real[:, :self.p, :].to(DEVICE)
        self.x_real_future = x_real[:, self.p:, :].to(DEVICE)

        self.res_estimate = lr_rsig(self.x_real_future, self.x_real_past, self.A1, self.A2, self.xi1, self.xi2, self.N,
                                    self.activation)[self.indices].clone()
        self.x_past_sample = self.x_real_past[self.indices].clone()

    def __call__(self, x_fake):
        if self.terminal:
            expected_reservoir_fake = compute_rsig_terminal(x_fake, self.A1, self.A2, self.xi1, self.xi2, self.N,
                                                            self.activation).mean(0)
        else:
            expected_reservoir_fake = compute_rsig(x_fake, self.A1, self.A2, self.xi1, self.xi2, self.N,
                                                   self.activation).mean(0)
        loss = torch.norm(self.res_estimate - expected_reservoir_fake, p=2, dim=1).mean()
        return loss


# -----------------------------------------------------------
# Defines the training procedure for the generator using
# the RSigCW1 metric as loss function
# -----------------------------------------------------------

class RSigCWGANTraining:
    def __init__(self, x_train, x_val, batch_size, generator, p, q, dim_res, mc_num, num_grad_steps, learning_rate,
                 activation, device=DEVICE, terminal=True):

        self.p = p
        self.q = q
        self.x_train = x_train
        self.x_val = x_val
        self.x_train_past = self.x_train[:, :self.p]
        self.x_train_future = self.x_train[:, self.p:]

        self.batch_size = batch_size
        self.best_generator = None
        self.generator = generator
        self.dim_res = dim_res
        self.mc_num = mc_num
        self.generator_optim = optim.Adam(self.generator.parameters())
        self.num_grad_steps = num_grad_steps
        self.learning_rate = learning_rate
        self.activation = activation

        self.A1, self.A2 = A1, A2

        self.xi1, self.xi2 = xi1, xi2

        self.train_losses_history = defaultdict(list)
        self.val_losses_history = defaultdict(list)
        self.device = device
        self.terminal = terminal

        self.res_estimate = lr_rsig(self.x_train_future, self.x_train_past, self.A1, self.A2, self.xi1, self.xi2,
                                    self.dim_res, self.activation)

        self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.generator_optim, gamma=0.95, step_size=128)
        self.best_loss = None

    def sample_batch(self):
        indices = sample_indices(self.res_estimate.shape[0], self.batch_size)
        rsig_pred = self.res_estimate[indices].clone().to(self.device)
        x_past = self.x_train_past[indices].clone().to(self.device)
        return rsig_pred, x_past

    def sample_rsig_fake(self, mc_batch_size=1000):
        x_past_mc = self.x_train_past.repeat(mc_batch_size, 1, 1).requires_grad_()
        x_fake = self.generator(mc_batch_size, self.q, x_past_mc)
        if self.terminal:
            rsig_fake_future = compute_rsig_terminal(x_fake, self.A1, self.A2, self.xi1, self.xi2, self.dim_res,
                                                     self.activation)
        else:
            rsig_fake_future = compute_rsig(x_fake, self.A1, self.A2, self.xi1, self.xi2, self.dim_res, self.activation)
        rsig_fake_ce = rsig_fake_future.reshape(mc_batch_size, self.x_train_past.size(0), -1).mean(0)
        return rsig_fake_ce, x_fake

    def fit(self):
        self.generator.to(self.device)

        for j in tqdm(range(self.num_grad_steps)):
            self.generator_optim.zero_grad()
            rsig_pred, x_past = self.sample_batch()
            x = torch.empty([self.mc_num, self.q, 1], device=self.device)
            for sample in x_past:
                x_fake = self.generator(self.mc_num, self.q, sample.reshape(1, self.p, 1)).to(self.device)
                x = torch.cat([x, x_fake], dim=0)
            if self.terminal:
                rsig_fake = compute_rsig_terminal(x[self.mc_num:, :, :], self.A1, self.A2, self.xi1, self.xi2,
                                                  self.dim_res, self.activation)
            else:
                rsig_fake = compute_rsig_terminal(x[self.mc_num:, :, :], self.A1, self.A2, self.xi1, self.xi2,
                                                  self.dim_res, self.activation)
            rsig_fake_mc = rsig_fake.reshape(self.mc_num, self.batch_size, self.dim_res).mean(0)

            loss = torch.norm(rsig_pred - rsig_fake_mc, p=2, dim=1).mean()
            loss.backward()
            self.best_loss = loss.item() if j == 0 else self.best_loss
            if (j + 1) % 100 == 0:
                print("rsig-c-w1 loss: {:1.2e}, best loss: {:1.2e}".format(loss.item(), self.best_loss))
            self.generator_optim.step()
            self.scheduler.step()
            self.train_losses_history["RSigCW1Loss"].append(loss.item())
            if loss < self.best_loss:
                self.best_generator = deepcopy(self.generator.state_dict())
                self.best_loss = loss

        self.generator.load_state_dict(self.best_generator)