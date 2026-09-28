from contextlib import contextmanager

import torch


def l2_dist(x, y: float) -> float:
    return (x - y).pow(2).sum().sqrt()


@contextmanager
def fixed_rng(seed: int):
    """
    Run a block under a fixed RNG stream and restore the training stream afterwards.

    Used so that checkpoint scoring draws the same randomness at every step, which
    makes scores comparable across steps without perturbing the training draws.
    """
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
