from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from fraud.training.features import FEATURE_COLUMNS, LABEL_COLUMN


@pytest.fixture
def make_synthetic_split() -> Callable[[int, int], tuple[pd.DataFrame, pd.Series]]:
    def factory(rows: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(seed)
        label = rng.integers(0, 2, size=rows)
        base = rng.normal(size=(rows, len(FEATURE_COLUMNS)))
        signal = label[:, None] * 3.0
        x = pd.DataFrame(base + signal, columns=list(FEATURE_COLUMNS))
        y = pd.Series(label, name=LABEL_COLUMN, dtype="int8")
        return x, y

    return factory
