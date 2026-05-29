import numpy as np
import pytest

from fraud.evaluation.business import CostMatrix
from fraud.evaluation.threshold import (
    ThresholdConstraints,
    select_threshold,
)


def _matrix() -> CostMatrix:
    return CostMatrix(fn_cost_usd=100.0, fp_cost_usd=5.0)


def _permissive() -> ThresholdConstraints:
    return ThresholdConstraints(recall_floor=0.0, alert_volume_budget=1.0)


def test_threshold_constraints_reject_recall_floor_out_of_range() -> None:
    with pytest.raises(ValueError, match="recall_floor"):
        ThresholdConstraints(recall_floor=1.5, alert_volume_budget=0.1)


def test_threshold_constraints_reject_alert_budget_out_of_range() -> None:
    with pytest.raises(ValueError, match="alert_volume_budget"):
        ThresholdConstraints(recall_floor=0.5, alert_volume_budget=-0.1)


def test_select_threshold_minimizes_cost_under_permissive_constraints() -> None:
    y = np.array([1, 1, 0, 0, 0, 1, 0])
    scores = np.array([0.95, 0.80, 0.10, 0.20, 0.30, 0.50, 0.70])

    decision = select_threshold(y, scores, matrix=_matrix(), constraints=_permissive())

    assert 0.0 <= decision.threshold <= 1.0
    assert decision.expected_cost >= 0.0
    assert decision.flagged_rate <= 1.0


def test_select_threshold_raises_when_recall_floor_infeasible() -> None:
    y = np.array([1, 1, 1, 0, 0])
    scores = np.array([0.1, 0.1, 0.1, 0.9, 0.9])
    constraints = ThresholdConstraints(recall_floor=0.9, alert_volume_budget=0.2)

    with pytest.raises(RuntimeError, match="no threshold"):
        select_threshold(y, scores, matrix=_matrix(), constraints=constraints)


def test_select_threshold_raises_when_alert_budget_infeasible() -> None:
    y = np.array([1, 0, 0, 0, 0])
    scores = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    constraints = ThresholdConstraints(recall_floor=1.0, alert_volume_budget=0.01)

    with pytest.raises(RuntimeError):
        select_threshold(y, scores, matrix=_matrix(), constraints=constraints)


def test_select_threshold_breaks_ties_with_highest_threshold() -> None:
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.6, 0.4, 0.1])
    matrix = CostMatrix(fn_cost_usd=0.0, fp_cost_usd=0.0)

    decision = select_threshold(y, scores, matrix=matrix, constraints=_permissive())

    assert decision.threshold == pytest.approx(0.9)
    assert decision.flagged_rate == pytest.approx(0.25)


def test_select_threshold_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        select_threshold(
            np.array([], dtype=int),
            np.array([], dtype=float),
            matrix=_matrix(),
            constraints=_permissive(),
        )


def test_select_threshold_rejects_nan_scores() -> None:
    y = np.array([1, 0, 1])
    scores = np.array([0.5, np.nan, 0.9])

    with pytest.raises(ValueError, match="finite"):
        select_threshold(y, scores, matrix=_matrix(), constraints=_permissive())
