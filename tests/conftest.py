from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from fraud.training.features import FEATURE_COLUMNS, LABEL_COLUMN

# Signal lives in a fixed handful of columns with deliberate overlap, so the synthetic
# model is strong but imperfect (non-zero cost) and stays that way as the contract widens.
SIGNAL_FEATURE_COUNT = 8
SIGNAL_STRENGTH = 1.2


@pytest.fixture
def make_synthetic_split() -> Callable[[int, int], tuple[pd.DataFrame, pd.Series]]:
    def factory(rows: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(seed)
        label = rng.integers(0, 2, size=rows)
        base = rng.normal(size=(rows, len(FEATURE_COLUMNS)))
        base[:, :SIGNAL_FEATURE_COUNT] += label[:, None] * SIGNAL_STRENGTH
        x = pd.DataFrame(base, columns=list(FEATURE_COLUMNS))
        y = pd.Series(label, name=LABEL_COLUMN, dtype="int8")
        return x, y

    return factory
