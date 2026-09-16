import torch


def generate_in_chunks(generator, draws_per_past: int, q: int, x_past: torch.tensor, chunk_size: int = None):
    """
    Generate draws_per_past futures for every past.

    The generator handles all pasts in one call; chunking only bounds peak memory, since
    n_past * draws_per_past paths are held at once. Chunks are contiguous in past order,
    so the [past_0 x draws, past_1 x draws, ...] grouping is preserved.
    """
    n_past = x_past.shape[0]
    if not chunk_size or chunk_size >= n_past:
        return generator(draws_per_past, q, x_past)
    blocks = [generator(draws_per_past, q, x_past[i:i + chunk_size])
              for i in range(0, n_past, chunk_size)]
    return torch.cat(blocks, dim=0)
