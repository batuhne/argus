"""Seed the standard, NumPy, and hash RNGs so a run is reproducible."""

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed the standard library and numpy RNGs for reproducible runs.

    Model libraries take the same seed through their own parameters at fit time.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
