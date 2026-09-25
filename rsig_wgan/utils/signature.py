import math
import torch


def _exp_levels(dx, depth):
    """Signature of a straight line with increment dx."""
    powers, cur = [dx], dx
    for _ in range(2, depth + 1):
        cur = (cur.unsqueeze(-1) * dx.unsqueeze(-2)).flatten(1)
        powers.append(cur)
    return [p / math.factorial(k) for k, p in enumerate(powers, start=1)]


def _chen(a, b, depth):
    """Chen's relation."""
    out = []
    for k in range(1, depth + 1):
        acc = a[k - 1] + b[k - 1]
        for i in range(1, k):
            acc = acc + (a[i - 1].unsqueeze(-1) * b[k - i - 1].unsqueeze(-2)).flatten(1)
        out.append(acc)
    return out


def signature(path, depth):
    """Compute (truncated) path signature."""
    increments = path[:, 1:] - path[:, :-1]
    sig = None
    for i in range(increments.shape[1]):
        step = _exp_levels(increments[:, i], depth)
        sig = step if sig is None else _chen(sig, step, depth)
    return torch.cat(sig, dim=1)
