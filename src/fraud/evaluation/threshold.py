from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fraud.evaluation._coercion import coerce_pair
from fraud.evaluation.business import CostMatrix


@dataclass(frozen=True, slots=True)
class ThresholdConstraints:
    recall_floor: float
    alert_volume_budget: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.recall_floor <= 1.0:
            raise ValueError(f"recall_floor must be in [0, 1], got {self.recall_floor}")
        if not 0.0 <= self.alert_volume_budget <= 1.0:
            raise ValueError(
                f"alert_volume_budget must be in [0, 1], got {self.alert_volume_budget}"
            )


@dataclass(frozen=True, slots=True)
class ThresholdDecision:
    threshold: float
    expected_cost: float
    expected_cost_per_tx: float
    recall: float
    precision: float
    flagged_rate: float


def select_threshold(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    matrix: CostMatrix,
    constraints: ThresholdConstraints,
) -> ThresholdDecision:
    """Pick the cost-minimizing threshold under the recall and alert-volume limits."""
    y_arr, score_arr = coerce_pair(y_true, y_score, require_finite=True)
    if len(y_arr) == 0:
        raise ValueError("cannot select a threshold from an empty evaluation set")

    curve = _cost_curve(y_arr, score_arr, matrix)
    feasible = _feasible_mask(curve, constraints)
    if not feasible.any():
        raise RuntimeError(
            "no threshold satisfies recall_floor"
            f"={constraints.recall_floor} and alert_volume_budget={constraints.alert_volume_budget}"
        )

    best_idx = _argmin_with_highest_threshold(curve, feasible)
    return _decision_at(curve, best_idx, n=len(y_arr))


@dataclass(frozen=True, slots=True)
class _CostCurve:
    thresholds: NDArray[np.float64]
    recall: NDArray[np.float64]
    precision: NDArray[np.float64]
    flagged_rate: NDArray[np.float64]
    total_cost: NDArray[np.float64]


def _cost_curve(
    y_true: NDArray[np.int8], y_score: NDArray[np.float64], matrix: CostMatrix
) -> _CostCurve:
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order].astype(np.int64, copy=False)
    scores_sorted = y_score[order]

    n = len(y_true)
    positives = int(y_sorted.sum())

    tp_cum = np.cumsum(y_sorted)
    flagged = np.arange(1, n + 1, dtype=np.int64)
    fp_cum = flagged - tp_cum

    boundary = _last_index_per_distinct_score(scores_sorted)
    tp_cum = tp_cum[boundary]
    flagged = flagged[boundary]
    fp_cum = fp_cum[boundary]
    thresholds = scores_sorted[boundary]

    recall = (tp_cum / positives) if positives > 0 else np.zeros(len(tp_cum), dtype=np.float64)
    precision = tp_cum / np.maximum(flagged, 1)
    flagged_rate = flagged / n
    fn_count = positives - tp_cum
    total_cost = fn_count * matrix.fn_cost_usd + fp_cum * matrix.fp_cost_usd

    return _CostCurve(
        thresholds=thresholds.astype(np.float64, copy=False),
        recall=recall.astype(np.float64, copy=False),
        precision=precision.astype(np.float64, copy=False),
        flagged_rate=flagged_rate.astype(np.float64, copy=False),
        total_cost=total_cost.astype(np.float64, copy=False),
    )


def _last_index_per_distinct_score(scores_desc: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Mark the last index of each tied-score run. Serving applies the threshold as
    score >= t, flagging every row that shares t rather than a rank-ordered prefix, so
    these are the only thresholds whose recall, flagged rate, and cost the >= rule reproduces."""
    boundary = np.ones(len(scores_desc), dtype=bool)
    boundary[:-1] = scores_desc[1:] != scores_desc[:-1]
    return boundary


def _feasible_mask(curve: _CostCurve, constraints: ThresholdConstraints) -> NDArray[np.bool_]:
    recall_ok = curve.recall >= constraints.recall_floor
    volume_ok = curve.flagged_rate <= constraints.alert_volume_budget
    return recall_ok & volume_ok


def _argmin_with_highest_threshold(curve: _CostCurve, feasible: NDArray[np.bool_]) -> int:
    """Min-cost feasible index; ties break toward fewer flags."""
    masked_cost = np.where(feasible, curve.total_cost, np.inf)
    min_cost = float(masked_cost.min())
    tie_mask = feasible & (curve.total_cost == min_cost)
    return int(np.flatnonzero(tie_mask)[0])


def _decision_at(curve: _CostCurve, idx: int, *, n: int) -> ThresholdDecision:
    return ThresholdDecision(
        threshold=float(curve.thresholds[idx]),
        expected_cost=float(curve.total_cost[idx]),
        expected_cost_per_tx=float(curve.total_cost[idx] / n) if n else math.nan,
        recall=float(curve.recall[idx]),
        precision=float(curve.precision[idx]),
        flagged_rate=float(curve.flagged_rate[idx]),
    )
