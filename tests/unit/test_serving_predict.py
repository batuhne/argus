from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from fraud.calibrator import IsotonicCalibrator
from fraud.model_loader import ModelBundle
from fraud.serving.predict import score_transaction
from fraud.transforms.encoders import CategoricalEncoder


class _ConstantModel:
    def __init__(self, positive_probability: float) -> None:
        self._p = positive_probability

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([[1.0 - self._p, self._p]])


def _identity_calibrator() -> IsotonicCalibrator:
    grid = np.array([0.0, 0.5, 1.0])
    regressor = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(grid, grid)
    return IsotonicCalibrator(regressor=regressor)


def _bundle(model: Any, threshold: float) -> ModelBundle:
    return ModelBundle(
        model=model,
        calibrator=_identity_calibrator(),
        encoder=CategoricalEncoder(columns=(), frequency_maps={}, target_maps={}, global_prior=0.0),
        threshold=threshold,
        version=1,
        family="lightgbm",
    )


def _features() -> pd.DataFrame:
    return pd.DataFrame([{"x": 0.0}])


def test_score_flags_when_calibrated_at_or_above_threshold() -> None:
    scored = score_transaction(_bundle(_ConstantModel(0.8), threshold=0.5), _features())
    assert scored.decision is True
    assert scored.fraud_score == pytest.approx(0.8, abs=1e-6)


def test_score_clears_when_calibrated_below_threshold() -> None:
    scored = score_transaction(_bundle(_ConstantModel(0.2), threshold=0.5), _features())
    assert scored.decision is False
    assert scored.fraud_score == pytest.approx(0.2, abs=1e-6)


def test_score_flags_exactly_at_threshold() -> None:
    scored = score_transaction(_bundle(_ConstantModel(0.5), threshold=0.5), _features())
    assert scored.decision is True


def test_score_echoes_bundle_threshold() -> None:
    scored = score_transaction(_bundle(_ConstantModel(0.3), threshold=0.42), _features())
    assert scored.threshold == pytest.approx(0.42)
