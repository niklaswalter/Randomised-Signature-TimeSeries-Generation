"""
Implements the Sig-Wasserstein-1 metric and the corresponding
training procedure of the generator

We use code from Ni et al. (2021), see GitHub:
https://github.com/SigCGANs/Sig-Wasserstein-GANs
"""

import signatory
import torch
from torch import optim
import math
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
from utils import *

"""
Time augmentation of input path
"""


def apply_time_augmentations(x: torch.tensor, device=DEVICE) -> torch.tensor:
    y = x.clone().to(device)
    t = torch.linspace(0, 1, y.shape[1]).reshape(1, -1, 1).repeat(y.shape[0], 1, 1).to(device)
    return torch.cat([t, x], dim=1)


"""
Basepoint augmentation of input path
"""


def apply_bp_augmentation(x: torch.tensor, device=DEVICE) -> torch.tensor:
    y = x.clone().to(device)
    basepoint = torch.zeros(y.shape[0], 1, y.shape[2]).to(device)
    return torch.cat([basepoint, x], dim=1)


"""
Visibility augmentation of input path
"""


def apply_ivisi_augmentation(x, device=DEVICE):
    y = x.clone().to(device)

    init_tworows_ = torch.zeros_like(y[:, :1, :]).to(device)
    init_tworows = torch.cat((init_tworows_, y[:, :1, :]), axis=1)

    temp = torch.cat((init_tworows, y), axis=1)

    last_col1 = torch.zeros_like(y[:, :2, :1]).to(device)
    last_col2 = torch.cat((last_col1, torch.ones_like(y[:, :, :1])), axis=1)

    output = torch.cat((temp, last_col2), axis=-1)
    return output


"""
Lead-lag transformation of input path
"""


def apply_lead_lag_augmentation(x: torch.tensor, device=DEVICE):
    y = x.clone().to(device)
    y_rep = torch.repeat_interleave(y, repeats=2, dim=1).to(device)
    y_lead_lag = torch.cat([y_rep[:, :-1], y_rep[:, 1:]], dim=2)
    return y_lead_lag


"""
Applying augmentations/transformations to input path
"""


def apply_augmentations(x: torch.tensor, time=True, lead_lag=True, ivisi=True, basepoint=False,
                        device=DEVICE) -> torch.tensor:
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


"""
Computing the expected signature of input paths 
"""


def compute_exp_sig(x: torch.tensor, trunc: int, augmented=True, normalise=True) -> torch.tensor:
    if augmented:
        x = apply_augmentations(x.clone()).to(DEVICE)
    exp_sig = signatory.signature(x, depth=trunc).mean(0).to(DEVICE)
    dim = x.shape[2]
    count = 0
    if normalise:
        for i in range(trunc):
            exp_sig[count:count + dim ** (i + 1)] = exp_sig[count:count + dim ** (i + 1)] * math.factorial(i + 1)
            count = count + dim ** (i + 1)
    return exp_sig.to(DEVICE)


class SigW1Metric:
    """
    Class for implementation of Sig-W1 metric
    """
    
    def __init__(self, x_real, trunc, normalise=True, device=DEVICE):
        self.x_real = x_real
        self.trunc = trunc
        self.normalise = normalise
        self.device = device
        self.name = "Sig-W1-Dist"

        self.exp_sig_real = compute_exp_sig(self.x_real, self.trunc, self.normalise).to(self.device)

    def __call__(self, x_fake):
        exp_sig_fake = compute_exp_sig(x_fake, self.trunc, self.normalise).to(self.device)
        return l2_dist(self.exp_sig_real.to(self.device), exp_sig_fake)


class SigWGANTraining:
    """
    Class for training procedure with Sig-W1 discriminator
    """

    def __init__(self, x_train, x_val, batch_size, generator, num_grad_steps, learning_rate, trunc, normalise=True,
                 device=DEVICE):
        self.x_train = x_train
        self.x_val = x_val
        self.batch_size = batch_size
        self.n_lags = self.x_train.shape[1]
        self.generator = generator
        self.generator_optim = optim.Adam(self.generator.parameters())
        self.best_generator = None
        self.num_grad_steps = num_grad_steps
        self.learning_rate = learning_rate
        self.trunc = trunc
        self.normalise = normalise
        self.device = device

        self.train_losses_history = defaultdict(list)
        self.val_losses_history = defaultdict(list)

        self.metric = SigW1Metric(self.x_train, trunc, self.normalise, self.device)
        self.metric_val = SigW1Metric(self.x_val, trunc, self.normalise, self.device)
        self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.generator_optim, gamma=0.95, step_size=128)

    """
    Method to fit model using Adam optimiser
    """

    def fit(self):
        self.generator.to(self.device)
        best_loss = None

        for j in tqdm(range(self.num_grad_steps)):
            self.generator_optim.zero_grad()
            x_fake = self.generator(batch_size=self.batch_size, n_lags=self.n_lags).to(self.device)
            loss = self.metric(x_fake)
            loss.backward()
            best_loss = loss.item() if j == 0 else best_loss
            if (j + 1) % 100 == 0:
                val_loss = self.metric_val(x_fake)
                self.val_losses_history["SigW1Val"].append(val_loss.item())
                print("sig-w1 - train loss: {:1.2e}, best train loss: {:1.2e}, val loss: {:1.2e}"
                      .format(loss.item(), best_loss, val_loss))
            self.generator_optim.step()
            self.scheduler.step()
            self.train_losses_history["SigW1Loss"].append(loss.item())
            if loss < best_loss:
                self.best_generator = deepcopy(self.generator.state_dict())
                best_loss = loss
        self.generator.load_state_dict(self.best_generator)
