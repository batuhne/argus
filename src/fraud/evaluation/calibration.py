from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    regressor: IsotonicRegression

    def predict(self, raw_scores: ArrayLike) -> NDArray[np.float64]:
        scores = np.asarray(raw_scores, dtype=np.float64)
        clipped: NDArray[np.float64] = np.clip(self.regressor.predict(scores), 0.0, 1.0)
        return clipped


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    calibrator: IsotonicCalibrator
    brier_before: float
    brier_after: float


def fit_isotonic(y_val: ArrayLike, raw_scores: ArrayLike) -> CalibrationResult:
    """Fit isotonic regression on validation scores; report Brier before and after."""
    y_arr = np.asarray(y_val).astype(np.int8, copy=False)
    score_arr = np.asarray(raw_scores, dtype=np.float64)
    _require_aligned(y_arr, score_arr)
    if not _has_both_classes(y_arr):
        raise ValueError("cannot fit isotonic calibration on a single-class validation set")

    regressor = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    regressor.fit(score_arr, y_arr)
    calibrator = IsotonicCalibrator(regressor=regressor)

    brier_before = float(brier_score_loss(y_arr, score_arr))
    brier_after = float(brier_score_loss(y_arr, calibrator.predict(score_arr)))
    return CalibrationResult(calibrator, brier_before, brier_after)


def reliability_curve_figure(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    n_bins: int = 10,
    title: str = "Reliability curve",
) -> Figure:
    """Diagonal-anchored reliability curve for the calibrated test scores."""
    from matplotlib.figure import Figure

    y_arr = np.asarray(y_true).astype(np.int8, copy=False)
    score_arr = np.asarray(y_score, dtype=np.float64)
    _require_aligned(y_arr, score_arr)

    figure = Figure(figsize=(6, 5))
    axes = figure.subplots()
    if _has_both_classes(y_arr):
        fraction_positive, mean_predicted = calibration_curve(y_arr, score_arr, n_bins=n_bins)
        axes.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1, label="perfect")
        axes.plot(mean_predicted, fraction_positive, marker="o", linewidth=1.5, label="model")
        axes.legend(loc="lower right")
    else:
        axes.text(0.5, 0.5, "reliability undefined (single class)", ha="center", va="center")
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.set_xlabel("Mean predicted probability")
    axes.set_ylabel("Empirical positive fraction")
    axes.set_title(title)
    figure.tight_layout()
    return figure


def _require_aligned(y_true: NDArray[np.int8], y_score: NDArray[np.float64]) -> None:
    if y_true.shape != y_score.shape:
        raise ValueError(f"y_true and y_score must align: got {y_true.shape} vs {y_score.shape}")


def _has_both_classes(y_true: NDArray[np.int8]) -> bool:
    total = int(y_true.sum())
    return 0 < total < len(y_true)
