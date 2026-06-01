import math

import pytest

from fraud.evaluation.business import CostMatrix
from fraud.monitoring.perf_monitor import RollingPerformance

MATRIX = CostMatrix(fn_cost_usd=100.0, fp_cost_usd=5.0)


def _monitor(**kwargs: int) -> RollingPerformance:
    return RollingPerformance(cost_matrix=MATRIX, **kwargs)


def test_score_then_label_matches() -> None:
    monitor = _monitor()
    monitor.observe_score("t-1", 0.9, decision=True)
    assert monitor.matched_count == 0
    assert monitor.pending_count == 1
    monitor.observe_label("t-1", 1)
    assert monitor.matched_count == 1
    assert monitor.pending_count == 0


def test_label_before_score_is_joined_when_score_arrives() -> None:
    monitor = _monitor()
    monitor.observe_label("t-1", 0)
    assert monitor.matched_count == 0
    monitor.observe_score("t-1", 0.1, decision=False)
    assert monitor.matched_count == 1


def test_duplicate_label_does_not_double_count() -> None:
    monitor = _monitor()
    monitor.observe_score("t-1", 0.9, decision=True)
    monitor.observe_label("t-1", 1)
    monitor.observe_label("t-1", 1)
    assert monitor.matched_count == 1


def test_redelivered_score_after_resolution_does_not_rejoin() -> None:
    monitor = _monitor()
    monitor.observe_score("t-1", 0.9, decision=True)
    monitor.observe_label("t-1", 1)
    monitor.observe_score("t-1", 0.9, decision=True)
    monitor.observe_label("t-1", 1)
    assert monitor.matched_count == 1
    assert monitor.pending_count == 0


def test_window_size_bounds_matched_history() -> None:
    monitor = _monitor(window_size=2)
    for i in range(5):
        tid = f"t-{i}"
        monitor.observe_score(tid, 0.5, decision=False)
        monitor.observe_label(tid, 0)
    assert monitor.matched_count == 2


def test_join_retention_evicts_oldest_unmatched_scores() -> None:
    monitor = _monitor(join_retention=2)
    for i in range(5):
        monitor.observe_score(f"t-{i}", 0.5, decision=False)
    assert monitor.pending_count == 2
    # The two survivors are the most recent; an old eviction never matches later.
    monitor.observe_label("t-0", 1)
    assert monitor.matched_count == 0


def test_business_cost_counts_false_negatives_and_positives() -> None:
    monitor = _monitor()
    monitor.observe_score("fn", 0.1, decision=False)
    monitor.observe_label("fn", 1)  # missed fraud -> fn_cost
    monitor.observe_score("fp", 0.9, decision=True)
    monitor.observe_label("fp", 0)  # false alarm -> fp_cost
    monitor.observe_score("tp", 0.9, decision=True)
    monitor.observe_label("tp", 1)  # caught fraud -> no cost
    assert monitor.business_cost_per_txn() == pytest.approx((100.0 + 5.0) / 3)
    assert monitor.flagged_rate() == pytest.approx(2 / 3)


def test_rolling_auprc_perfect_separation_is_one() -> None:
    monitor = _monitor()
    for i in range(4):
        monitor.observe_score(f"neg-{i}", 0.1, decision=False)
        monitor.observe_label(f"neg-{i}", 0)
    for i in range(4):
        monitor.observe_score(f"pos-{i}", 0.9, decision=True)
        monitor.observe_label(f"pos-{i}", 1)
    assert monitor.rolling_auprc() == pytest.approx(1.0)


def test_metrics_are_nan_when_empty() -> None:
    monitor = _monitor()
    assert math.isnan(monitor.rolling_auprc())
    assert math.isnan(monitor.business_cost_per_txn())
    assert math.isnan(monitor.flagged_rate())
