import numpy as np
import pandas as pd
import pytest

from fraud.training.features import FEATURE_COLUMNS
from fraud.training.models import BoostingHyperparams
from fraud.training.tune import tune_xgb


def _separable_dataset(rows: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, size=rows)
    base = rng.normal(size=(rows, len(FEATURE_COLUMNS)))
    signal = label[:, None] * 3.0  # strong signal so two trials converge fast
    frame = pd.DataFrame(base + signal, columns=list(FEATURE_COLUMNS))
    return frame, pd.Series(label, name="isFraud")


def test_tune_xgb_returns_typed_params_and_positive_auprc() -> None:
    x_train, y_train = _separable_dataset(rows=300, seed=1)
    x_val, y_val = _separable_dataset(rows=150, seed=2)

    best, value = tune_xgb(x_train, y_train, x_val, y_val, n_trials=2, timeout=None, seed=11)

    assert isinstance(best, BoostingHyperparams)
    assert 200 <= best.n_estimators <= 1000
    assert value > 0.7


def test_tune_xgb_raises_when_no_trial_completes() -> None:
    x_train, y_train = _separable_dataset(rows=50, seed=3)
    x_val, y_val = _separable_dataset(rows=20, seed=4)

    with pytest.raises(RuntimeError, match="no successful Optuna trials"):
        tune_xgb(x_train, y_train, x_val, y_val, n_trials=0, timeout=None, seed=11)
