"""Isotonic probability calibration and its reliability diagram."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from fraud.calibrator import IsotonicCalibrator
from fraud.evaluation._coercion import coerce_pair, has_both_classes

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    calibrator: IsotonicCalibrator


def fit_isotonic(y_val: ArrayLike, raw_scores: ArrayLike) -> CalibrationResult:
    """Fit isotonic regression on validation scores and return the calibrator."""
    y_arr, score_arr = coerce_pair(y_val, raw_scores)
    if not has_both_classes(y_arr):
        raise ValueError("cannot fit isotonic calibration on a single-class validation set")

    regressor = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    regressor.fit(score_arr, y_arr)
    return CalibrationResult(IsotonicCalibrator(regressor=regressor))


def brier_score(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Brier score; evaluate on held-out data since isotonic interpolates its own fit data."""
    y_arr, score_arr = coerce_pair(y_true, y_score, require_finite=True)
    return float(brier_score_loss(y_arr, score_arr))


def reliability_curve_figure(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    n_bins: int = 10,
    title: str = "Reliability curve",
) -> Figure:
    """Diagonal-anchored reliability curve for the calibrated test scores."""
    from matplotlib.figure import Figure

    y_arr, score_arr = coerce_pair(y_true, y_score)

    figure = Figure(figsize=(6, 5))
    axes = figure.subplots()
    if has_both_classes(y_arr):
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
