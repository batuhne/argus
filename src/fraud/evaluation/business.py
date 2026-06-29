"""Cost model: confusion counts and expected USD cost at a decision threshold."""

from __future__ import annotations

import math
from dataclasses import dataclass

from numpy.typing import ArrayLike

from fraud.evaluation._coercion import coerce_pair


@dataclass(frozen=True, slots=True)
class CostMatrix:
    fn_cost_usd: float
    fp_cost_usd: float

    def __post_init__(self) -> None:
        if self.fn_cost_usd < 0.0:
            raise ValueError(f"fn_cost_usd must be non-negative, got {self.fn_cost_usd}")
        if self.fp_cost_usd < 0.0:
            raise ValueError(f"fp_cost_usd must be non-negative, got {self.fp_cost_usd}")


def confusion_at_threshold(
    y_true: ArrayLike, y_score: ArrayLike, threshold: float
) -> tuple[int, int, int, int]:
    """Return (true_pos, false_pos, true_neg, false_neg) at the given threshold."""
    y_true_arr, y_score_arr = coerce_pair(y_true, y_score)
    flagged = y_score_arr >= threshold
    positives = int(y_true_arr.sum())
    negatives = len(y_true_arr) - positives
    true_pos = int(y_true_arr[flagged].sum())
    false_pos = int(flagged.sum() - true_pos)
    false_neg = positives - true_pos
    true_neg = negatives - false_pos
    return true_pos, false_pos, true_neg, false_neg


def expected_cost(
    y_true: ArrayLike, y_score: ArrayLike, threshold: float, matrix: CostMatrix
) -> float:
    """Total USD cost over the evaluation set: FN cost plus FP cost at threshold."""
    _, false_pos, _, false_neg = confusion_at_threshold(y_true, y_score, threshold)
    return float(false_neg * matrix.fn_cost_usd + false_pos * matrix.fp_cost_usd)


def expected_cost_per_transaction(
    y_true: ArrayLike, y_score: ArrayLike, threshold: float, matrix: CostMatrix
) -> float:
    """Per-transaction expected cost; NaN on empty input."""
    y_true_arr, _ = coerce_pair(y_true, y_score)
    n = len(y_true_arr)
    if n == 0:
        return math.nan
    return expected_cost(y_true, y_score, threshold, matrix) / n
