import torch
import numpy as np
from rsig_wgan.data import to_numpy
from rsig_wgan.discriminator_models import l2_dist

def cov(x, rowvar=False, bias=True, ddof=None, aweights=None):
    x = to_numpy(x)
    _, L, C = x.shape
    x = x.reshape(-1, L*C)
    return torch.from_numpy(np.cov(x, rowvar=False)).float()


def cov_diff(x_real, x_fake):
    cov_real, cov_fake = cov(x_real), cov(x_fake)
    return torch.norm(cov_real - cov_fake, p = 'fro')


def acf(x, lag, dim=(0, 1)):
    acf_list = list()
    x = x - x.mean((0, 1))
    std = torch.var(x, unbiased=False, dim=(0, 1))
    for i in range(lag):
        y = x[:, i:] * x[:, :-i] if i > 0 else torch.pow(x, 2)
        acf_i = torch.mean(y, dim) / std
        acf_list.append(acf_i)
    if dim == (0, 1):
        return torch.stack(acf_list)
    else:
        return torch.cat(acf_list, 1)


def acf_diff(x_real, x_fake, lag, dim=(0, 1)):
    return l2_dist(acf(x_real, lag), acf(x_fake, lag))