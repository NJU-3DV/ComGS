import random

import numpy as np
import torch


def set_global_seed(seed):
    if seed is None:
        return

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_numpy_rng(seed):
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


def make_torch_generator(seed, device):
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(int(seed))
    return generator
