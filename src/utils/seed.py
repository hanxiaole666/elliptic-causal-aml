"""Deterministic seeding across all random sources used by the pipeline."""
from __future__ import annotations

import os
import random
import zlib

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import torch_geometric

        torch_geometric.seed_everything(seed)
    except Exception:
        pass
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def rng_for(seed: int, tag: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(tag.encode("utf-8"))])
