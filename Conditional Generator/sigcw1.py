"""
Implements the conditional Sig-Wasserstein-1 metric and the corresponding
training procedure of the generator

We use code from Liao et al. (2023), see GitHub:
https://github.com/SigCGANs/Conditional-Sig-Wasserstein-GANs
"""


import signatory
import torch
from torch import optim
import math
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
import sklearn
from sklearn.linear_model import LinearRegression
from utils import *


def apply_time_augmentations(x, device=DEVICE):
    y = x.clone().to(device)
    t = torch.linspace(0, 1, y.shape[1]).reshape(1, -1, 1).repeat(y.shape[0], 1, 1).to(device)
    return torch.cat([t, x], dim=1)


def apply_bp_augmentation(x, device=DEVICE):
    y = x.clone().to(device)
    basepoint = torch.zeros(y.shape[0], 1, y.shape[2]).to(device)
    return torch.cat([basepoint, x], dim=1)


def apply_ivisi_augmentation(x, device=DEVICE):
    y = x.clone().to(device)

    init_tworows_ = torch.zeros_like(y[:, :1, :]).to(device)
    init_tworows = torch.cat((init_tworows_, y[:, :1, :]), axis=1)

    temp = torch.cat((init_tworows, y), axis=1)

    last_col1 = torch.zeros_like(y[:, :2, :1]).to(device)
    last_col2 = torch.cat((last_col1, torch.ones_like(y[:, :, :1])), axis=1)

    output = torch.cat((temp, last_col2), axis=-1)
    return output


def apply_lead_lag_augmentation(x, device=DEVICE):
    y = x.clone().to(device)
    y_rep = torch.repeat_interleave(y, repeats=2, dim=1).to(device)
    y_lead_lag = torch.cat([y_rep[:, :-1], y_rep[:, 1:]], dim=2)
    return y_lead_lag


def apply_augmentations(x, time=True, lead_lag=True, ivisi=True, basepoint=False, device=DEVICE):
    y = x.clone().to(device)
    if time:
        y = apply_time_augmentations(y, device)
    if lead_lag:
        y = apply_lead_lag_augmentation(y, device)
    if ivisi:
        y = apply_ivisi_augmentation(y, device)
    if basepoint:
        y = apply_bp_augmentation(y, device)
    return y


def lr_sig(x_future: torch.tensor, x_past: torch.tensor, trunc: int, augmented: bool = True):
    if augmented:
        x_future = apply_augmentations(x_future.clone()).to(DEVICE)
        x_past = apply_augmentations(x_past.clone()).to(DEVICE)
    sig_future = signatory.signature(x_future, depth=trunc)
    sig_past = signatory.signature(x_past, depth=trunc)
    X, y = to_numpy(sig_past), to_numpy(sig_future)
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X, y)
    return torch.from_numpy(lr.predict(X)).float()


class SigCW1Metric:
    def __init__(self, x_real, p, q, indices, trunc, augmented: bool = True, device=DEVICE):
        self.indices = indices
        self.device = device
        self.trunc = trunc
        self.augmented = augmented

        self.name = "Sig-Cond-W1-Dist"

        self.x_real = x_real
        self.n_lags = self.x_real.shape[1]
        self.p = p
        self.q = q
        self.x_real_past = x_real[:, :self.p, :]
        self.x_real_future = x_real[:, self.p:, :]

        self.sig_estimate = lr_sig(self.x_real_future, self.x_real_past, self.trunc, self.augmented)[
            self.indices].clone()
        self.x_past_sample = self.x_real_past[self.indices].clone()

    def __call__(self, x_fake):
        expected_signature_fake = signatory.signature(x_fake, depth=TRUNC).mean(0)
        loss = torch.norm(self.sig_estimate - expected_signature_fake, p=2, dim=1).mean()
        return loss


class SigCWGANTraining:
    def __init__(self, x_train, x_val, batch_size, generator, p, q, mc_num, num_grad_steps, learning_rate,
                 trunc, augmented=True, device=DEVICE):

        self.p = p
        self.q = q
        self.x_train = x_train
        self.x_val = x_val
        self.x_train_past = self.x_train[:, :self.p]
        self.x_train_future = self.x_train[:, self.p:]
        self.augmented = augmented
        self.trunc = trunc

        self.batch_size = batch_size
        self.best_generator = None
        self.generator = generator
        self.mc_num = mc_num
        self.generator_optim = optim.Adam(self.generator.parameters())
        self.num_grad_steps = num_grad_steps
        self.learning_rate = learning_rate

        self.train_losses_history = defaultdict(list)
        self.val_losses_history = defaultdict(list)
        self.device = device

        self.sig_estimate = lr_sig(self.x_train_future, self.x_train_past, self.trunc, self.augmented)

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
        sig_fake_future = signatory.signature(x_fake, depth=self.trunc)
        sig_fake_ce = sig_fake_future.reshape(mc_batch_size, self.x_train_past.size(0), -1).mean(0)
        return sig_fake_ce, x_fake

    def fit(self):
        self.generator.to(self.device)

        for j in tqdm(range(self.num_grad_steps)):
            self.generator_optim.zero_grad()
            sig_pred, x_past = self.sample_batch()
            x = torch.empty([self.mc_num, self.q, 1], device=self.device)
            for sample in x_past:
                x_fake = self.generator(self.mc_num, self.q, sample.reshape(1, self.p, 1)).to(self.device)
                x = torch.cat([x, x_fake], dim=0)

            x_aug = apply_augmentations(x[self.mc_num:].clone()).to(self.device)

            sig_fake = signatory.signature(x_aug, depth=self.trunc)

            sig_fake_mc = sig_fake.reshape(self.mc_num, self.batch_size, TRUNC * 30).mean(0)

            loss = torch.norm(sig_pred - sig_fake_mc, p=2, dim=1).mean()
            loss.backward()
            self.best_loss = loss.item() if j == 0 else self.best_loss
            if (j + 1) % 100 == 0:
                print("sig-c-w1 loss: {:1.2e}, best loss: {:1.2e}".format(loss.item(), self.best_loss))
            self.generator_optim.step()
            self.scheduler.step()
            self.train_losses_history["SigCW1Loss"].append(loss.item())
            if loss < self.best_loss:
                self.best_generator = deepcopy(self.generator.state_dict())
                self.best_loss = loss

        self.generator.load_state_dict(self.best_generator)
