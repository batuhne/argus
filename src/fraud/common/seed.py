"""Seed the standard library and NumPy RNGs so a run is reproducible."""

import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
