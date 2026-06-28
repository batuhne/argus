from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score, precision_recall_curve

from fraud.evaluation._coercion import coerce_pair

if TYPE_CHECKING:
    from matplotlib.figure import Figure

IntArray = NDArray[np.int8]


def auprc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Average precision; returns NaN if y_true is single-class."""
    y_true_arr, y_score_arr = coerce_pair(y_true, y_score)
    if not _has_both_classes(y_true_arr):
        return math.nan
    return float(average_precision_score(y_true_arr, y_score_arr))


def recall_at_k(y_true: ArrayLike, y_score: ArrayLike, k: float) -> float:
    """Recall over the top k fraction; NaN if no positives or k outside (0, 1]."""
    if not 0.0 < k <= 1.0:
        return math.nan
    y_true_arr, y_score_arr = coerce_pair(y_true, y_score)
    positives = int(y_true_arr.sum())
    if positives == 0:
        return math.nan
    flagged = max(1, math.ceil(len(y_score_arr) * k))
    top_idx = np.argpartition(-y_score_arr, flagged - 1)[:flagged]
    return float(y_true_arr[top_idx].sum() / positives)


def classification_at_threshold(
    y_true: ArrayLike, y_score: ArrayLike, threshold: float
) -> dict[str, float]:
    y_true_arr, y_score_arr = coerce_pair(y_true, y_score)
    flagged = y_score_arr >= threshold
    positives = int(y_true_arr.sum())
    negatives = len(y_true_arr) - positives
    true_pos = int(y_true_arr[flagged].sum())
    false_pos = int(flagged.sum() - true_pos)
    return {
        "precision": _safe_ratio(true_pos, int(flagged.sum())),
        "recall": _safe_ratio(true_pos, positives),
        "false_positive_rate": _safe_ratio(false_pos, negatives),
        "flagged_rate": float(flagged.mean()) if len(flagged) else math.nan,
    }


def pr_curve_figure(y_true: ArrayLike, y_score: ArrayLike, title: str = "PR curve") -> Figure:
    from matplotlib.figure import Figure

    y_true_arr, y_score_arr = coerce_pair(y_true, y_score)
    figure = Figure(figsize=(6, 5))
    axes = figure.subplots()
    if _has_both_classes(y_true_arr):
        precision, recall, _ = precision_recall_curve(y_true_arr, y_score_arr)
        axes.plot(recall, precision, linewidth=1.5)
        axes.set_xlim(0.0, 1.0)
        axes.set_ylim(0.0, 1.0)
    else:
        axes.text(0.5, 0.5, "PR curve undefined (single class)", ha="center", va="center")
    axes.set_xlabel("Recall")
    axes.set_ylabel("Precision")
    axes.set_title(title)
    figure.tight_layout()
    return figure


def _has_both_classes(y_true: IntArray) -> bool:
    total = int(y_true.sum())
    return 0 < total < len(y_true)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan
