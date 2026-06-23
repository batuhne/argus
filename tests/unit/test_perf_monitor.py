import math

import pytest

from fraud.evaluation.business import CostMatrix
from fraud.monitoring.perf_monitor import (
    DEFAULT_RETENTION_SECONDS,
    DEFAULT_WINDOW_SIZE,
    RollingPerformance,
)

MATRIX = CostMatrix(fn_cost_usd=100.0, fp_cost_usd=5.0)


def _monitor(
    window_size: int = DEFAULT_WINDOW_SIZE,
    retention_seconds: float = DEFAULT_RETENTION_SECONDS,
) -> RollingPerformance:
    return RollingPerformance(
        cost_matrix=MATRIX, window_size=window_size, retention_seconds=retention_seconds
    )


def test_score_then_label_matches() -> None:
    monitor = _monitor()
    monitor.observe_score("t-1", 0.9, decision=True, event_time=0.0)
    assert monitor.matched_count == 0
    assert monitor.pending_count == 1
    monitor.observe_label("t-1", 1, event_time=0.0)
    assert monitor.matched_count == 1
    assert monitor.pending_count == 0


def test_label_before_score_is_joined_when_score_arrives() -> None:
    monitor = _monitor()
    monitor.observe_label("t-1", 0, event_time=0.0)
    assert monitor.matched_count == 0
    monitor.observe_score("t-1", 0.1, decision=False, event_time=0.0)
    assert monitor.matched_count == 1


def test_duplicate_label_does_not_double_count() -> None:
    monitor = _monitor()
    monitor.observe_score("t-1", 0.9, decision=True, event_time=0.0)
    monitor.observe_label("t-1", 1, event_time=0.0)
    monitor.observe_label("t-1", 1, event_time=0.0)
    assert monitor.matched_count == 1


def test_redelivery_within_retention_does_not_rejoin() -> None:
    monitor = _monitor(retention_seconds=100.0)
    monitor.observe_score("t-1", 0.9, decision=True, event_time=0.0)
    monitor.observe_label("t-1", 1, event_time=0.0)
    monitor.observe_score("t-1", 0.9, decision=True, event_time=1.0)
    monitor.observe_label("t-1", 1, event_time=1.0)
    assert monitor.matched_count == 1
    assert monitor.pending_count == 0


def test_redelivery_past_retention_does_not_double_count() -> None:
    monitor = _monitor(retention_seconds=2.0)
    monitor.observe_score("t-1", 0.9, decision=True, event_time=0.0)
    monitor.observe_label("t-1", 1, event_time=0.0)
    assert monitor.matched_count == 1
    # advance the clock past retention to evict t-1 from the dedup set
    monitor.observe_score("t-2", 0.1, decision=False, event_time=10.0)
    monitor.observe_label("t-2", 0, event_time=10.0)
    # stale redelivery: must not rejoin
    monitor.observe_score("t-1", 0.9, decision=True, event_time=0.0)
    monitor.observe_label("t-1", 1, event_time=0.0)
    assert monitor.matched_count == 2
    assert monitor.pending_count == 0


def test_stale_event_outside_retention_is_rejected() -> None:
    monitor = _monitor(retention_seconds=2.0)
    # anchor holds the front and carries the clock to 1000
    monitor.observe_score("anchor", 0.5, decision=False, event_time=1000.0)
    monitor.observe_score("stale", 0.9, decision=True, event_time=0.0)
    monitor.observe_label("stale", 1, event_time=0.0)
    assert monitor.matched_count == 0
    assert monitor.pending_count == 1


def test_window_size_bounds_matched_history() -> None:
    monitor = _monitor(window_size=2)
    for i in range(5):
        tid = f"t-{i}"
        monitor.observe_score(tid, 0.5, decision=False, event_time=float(i))
        monitor.observe_label(tid, 0, event_time=float(i))
    assert monitor.matched_count == 2


def test_pending_score_survives_until_its_late_label_within_retention() -> None:
    monitor = _monitor(retention_seconds=100.0)
    monitor.observe_score("t-1", 0.5, decision=False, event_time=0.0)
    # later scores stay within retention, so t-1 survives
    for i in range(2, 50):
        monitor.observe_score(f"t-{i}", 0.5, decision=False, event_time=float(i))
    monitor.observe_label("t-1", 1, event_time=60.0)
    assert monitor.matched_count == 1


def test_pending_score_evicted_once_event_time_passes_retention() -> None:
    monitor = _monitor(retention_seconds=2.0)
    monitor.observe_score("t-old", 0.5, decision=False, event_time=0.0)
    # clock passes retention before the label, so the score ages out
    monitor.observe_score("t-new", 0.5, decision=False, event_time=10.0)
    monitor.observe_label("t-old", 1, event_time=10.0)
    assert monitor.matched_count == 0


def test_business_cost_counts_false_negatives_and_positives() -> None:
    monitor = _monitor()
    monitor.observe_score("fn", 0.1, decision=False, event_time=0.0)
    monitor.observe_label("fn", 1, event_time=0.0)  # missed fraud -> fn_cost
    monitor.observe_score("fp", 0.9, decision=True, event_time=0.0)
    monitor.observe_label("fp", 0, event_time=0.0)  # false alarm -> fp_cost
    monitor.observe_score("tp", 0.9, decision=True, event_time=0.0)
    monitor.observe_label("tp", 1, event_time=0.0)  # caught fraud -> no cost
    assert monitor.business_cost_per_txn() == pytest.approx((100.0 + 5.0) / 3)
    assert monitor.flagged_rate() == pytest.approx(2 / 3)


def test_rolling_auprc_perfect_separation_is_one() -> None:
    monitor = _monitor()
    for i in range(4):
        monitor.observe_score(f"neg-{i}", 0.1, decision=False, event_time=0.0)
        monitor.observe_label(f"neg-{i}", 0, event_time=0.0)
    for i in range(4):
        monitor.observe_score(f"pos-{i}", 0.9, decision=True, event_time=0.0)
        monitor.observe_label(f"pos-{i}", 1, event_time=0.0)
    assert monitor.rolling_auprc() == pytest.approx(1.0)


def test_metrics_are_nan_when_empty() -> None:
    monitor = _monitor()
    assert math.isnan(monitor.rolling_auprc())
    assert math.isnan(monitor.business_cost_per_txn())
    assert math.isnan(monitor.flagged_rate())
