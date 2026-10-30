import random

import numpy as np
import torch

DEFAULT_SEED = 42


def set_seed(seed=DEFAULT_SEED):
    """Seed python, numpy, torch (CPU) and all CUDA devices; make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # True FP32 arithmetic on Ampere GPUs (the fault model is FP32); see DECISIONS.md
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_generator(seed=DEFAULT_SEED):
    """Dedicated generator for a DataLoader (shuffle order and worker base seeds)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id):
    """worker_init_fn: derive numpy/python seeds from the per-worker torch seed."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
