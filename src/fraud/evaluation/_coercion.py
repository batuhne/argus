"""Coerce a label/score pair into aligned int8 and float64 arrays."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def coerce_pair(
    y_true: ArrayLike, y_score: ArrayLike, *, require_finite: bool = False
) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
    """Align a label/score pair into int8 and float64 arrays of equal shape."""
    y_true_arr = np.asarray(y_true).astype(np.int8, copy=False)
    y_score_arr = np.asarray(y_score).astype(np.float64, copy=False)
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError(
            f"y_true and y_score must align: got {y_true_arr.shape} vs {y_score_arr.shape}"
        )
    if require_finite and not np.isfinite(y_score_arr).all():
        raise ValueError("y_score must be finite (no NaN or inf)")
    return y_true_arr, y_score_arr
