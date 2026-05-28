from collections.abc import Callable

import pandas as pd
import pytest

from fraud.training.models import BoostingHyperparams
from fraud.training.tune import tune_xgb

SyntheticSplit = Callable[[int, int], tuple[pd.DataFrame, pd.Series]]


def test_tune_xgb_returns_typed_params_and_positive_auprc(
    make_synthetic_split: SyntheticSplit,
) -> None:
    x_train, y_train = make_synthetic_split(300, 1)
    x_val, y_val = make_synthetic_split(150, 2)

    best, value = tune_xgb(x_train, y_train, x_val, y_val, n_trials=2, timeout=None, seed=11)

    assert isinstance(best, BoostingHyperparams)
    assert 200 <= best.n_estimators <= 1000
    assert value > 0.7


def test_tune_xgb_raises_when_no_trial_completes(make_synthetic_split: SyntheticSplit) -> None:
    x_train, y_train = make_synthetic_split(50, 3)
    x_val, y_val = make_synthetic_split(20, 4)

    with pytest.raises(RuntimeError, match="no successful Optuna trials"):
        tune_xgb(x_train, y_train, x_val, y_val, n_trials=0, timeout=None, seed=11)
