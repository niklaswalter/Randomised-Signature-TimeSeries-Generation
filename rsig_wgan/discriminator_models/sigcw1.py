"""
Implements the conditional Sig-Wasserstein-1 metric and the corresponding
training procedure of the generator

We use code from Liao et al. (2023), see GitHub:
https://github.com/SigCGANs/Conditional-Sig-Wasserstein-GANs
"""

from collections import defaultdict
from copy import deepcopy

import torch
from sklearn.linear_model import LinearRegression
from torch import optim
from tqdm import tqdm

from rsig_wgan.utils import generate_in_chunks, sample_indices, signature, to_numpy

from .sigw1 import apply_augmentations


def compute_sig(x: torch.tensor, trunc: int, device: str, augmented: bool = True) -> torch.tensor:
    if augmented:
        x = apply_augmentations(x, device)
    return signature(x.to(device), trunc)


def fit_lr_sig(
    x_future: torch.tensor,
    x_past: torch.tensor,
    trunc: int,
    device: str,
    augmented: bool = True
) -> LinearRegression:
    """
    Fit the conditional expectation of the future signature given the past one
    """
    X = to_numpy(compute_sig(x_past, trunc, device, augmented))
    y = to_numpy(compute_sig(x_future, trunc, device, augmented))
    estimator = LinearRegression(fit_intercept=True)
    estimator.fit(X, y)
    return estimator


def predict_lr_sig(
    estimator: LinearRegression,
    x_past: torch.tensor,
    trunc: int,
    device: str,
    augmented: bool = True
) -> torch.tensor:
    """
    Apply a fitted estimator to pasts it was not fitted on
    """
    X = to_numpy(compute_sig(x_past, trunc, device, augmented))
    return torch.from_numpy(estimator.predict(X)).float()


def lr_sig(
    x_future: torch.tensor,
    x_past: torch.tensor,
    trunc: int,
    device: str,
    augmented: bool = True
) -> torch.tensor:
    estimator = fit_lr_sig(x_future, x_past, trunc, device, augmented)
    return predict_lr_sig(estimator, x_past, trunc, device, augmented)


class SigCW1Metric:
    def __init__(
        self,
        x_real: torch.tensor,
        indices,
        p: int,
        q: int,
        trunc: int,
        device: str,
        augmented: bool = True
    ):
        self.indices = indices
        self.trunc = trunc
        self.augmented = augmented
        self.device = device

        self.name = "Sig-Cond-W1-Dist"

        self.x_real = x_real
        self.n_lags = self.x_real.shape[1]
        self.p = p
        self.q = q
        self.x_real_past = x_real[:, :self.p, :].to(self.device)
        self.x_real_future = x_real[:, self.p:, :].to(self.device)

        self.sig_estimate = lr_sig(
            self.x_real_future, self.x_real_past, self.trunc, self.device, self.augmented
        )[self.indices].clone()
        self.x_past_sample = self.x_real_past[self.indices].clone()

    def __call__(self, x_fake: torch.tensor) -> torch.tensor:
        expected_signature_fake = compute_sig(x_fake, self.trunc, self.device, self.augmented).mean(0)
        return torch.norm(self.sig_estimate - expected_signature_fake, p=2, dim=1).mean()


class SigCWGANTraining:
    """
    Class for training procedure with Sig-CW1 discriminator
    """

    def __init__(self, x_train, x_val, batch_size, generator, p, q, mc_num, num_grad_steps, learning_rate,
                 trunc, device, augmented=True, past_chunk=None):

        self.p = p
        self.q = q
        self.x_train = x_train
        self.x_val = x_val
        self.x_train_past = self.x_train[:, :self.p]
        self.x_train_future = self.x_train[:, self.p:]

        self.batch_size = batch_size
        self.best_generator = None
        self.generator = generator
        self.mc_num = mc_num
        self.num_grad_steps = num_grad_steps
        self.learning_rate = learning_rate
        self.generator_optim = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.trunc = trunc
        self.augmented = augmented
        self.past_chunk = past_chunk

        self.train_losses_history = defaultdict(list)
        self.val_losses_history = defaultdict(list)
        self.device = device

        self.sig_estimate = lr_sig(
            self.x_train_future, self.x_train_past, self.trunc, self.device, self.augmented
        )

        self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.generator_optim, gamma=0.95, step_size=128)
        self.best_loss = None

    def sample_batch(self):
        indices = sample_indices(self.sig_estimate.shape[0], self.batch_size)
        sig_pred = self.sig_estimate[indices].clone().to(self.device)
        x_past = self.x_train_past[indices].clone().to(self.device)
        return sig_pred, x_past

    def sample_sig_fake(self, mc_batch_size=1000):
        x_past_mc = self.x_train_past.repeat(mc_batch_size, 1, 1).requires_grad_()
        x_fake = self.generator(mc_batch_size, self.q, x_past_mc)
        sig_fake_future = compute_sig(x_fake, self.trunc, self.device, self.augmented)
        sig_fake_ce = sig_fake_future.reshape(mc_batch_size, self.x_train_past.size(0), -1).mean(0)
        return sig_fake_ce, x_fake

    def fit(self):
        self.generator.to(self.device)

        for j in tqdm(range(self.num_grad_steps)):
            self.generator_optim.zero_grad()
            sig_pred, x_past = self.sample_batch()
            x = generate_in_chunks(self.generator, self.mc_num, self.q, x_past, self.past_chunk).to(self.device)

            sig_fake = compute_sig(x, self.trunc, self.device, self.augmented)
            sig_fake_mc = sig_fake.reshape(self.batch_size, self.mc_num, -1).mean(1)

            loss = torch.norm(sig_pred - sig_fake_mc, p=2, dim=1).mean()
            loss.backward()
            if j == 0:
                self.best_loss = loss.item()
                self.best_generator = deepcopy(self.generator.state_dict())
            if (j + 1) % 100 == 0:
                print("sig-c-w1 loss: {:1.2e}, best loss: {:1.2e}".format(loss.item(), self.best_loss))
            self.generator_optim.step()
            self.scheduler.step()
            self.train_losses_history["SigCW1Loss"].append(loss.item())
            if loss < self.best_loss:
                self.best_generator = deepcopy(self.generator.state_dict())
                self.best_loss = loss

        self.generator.load_state_dict(self.best_generator)
