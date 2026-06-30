"""Isotonic calibrator loaded by serving alongside the model.

Lives outside the evaluation layer so the serving path imports it without pulling evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    regressor: IsotonicRegression

    def predict(self, raw_scores: ArrayLike) -> NDArray[np.float64]:
        scores = np.asarray(raw_scores, dtype=np.float64)
        clipped: NDArray[np.float64] = np.clip(self.regressor.predict(scores), 0.0, 1.0)
        return clipped
