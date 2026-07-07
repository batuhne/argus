"""Rolling model performance from an event-time-bounded join of scores and delayed labels."""

from __future__ import annotations

import math
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from fraud.evaluation.business import CostMatrix
from fraud.evaluation.metrics import auprc

DEFAULT_WINDOW_SIZE = 5000
# Seconds a score stays joinable, set above the longest label lag; also the restart-replay window.
DEFAULT_RETENTION_SECONDS = 1800.0
# Hard ceiling on tracked ids: retention eviction is the primary bound, but a stalled event
# clock never advances the cutoff, so this backstops memory against unbounded growth.
DEFAULT_MAX_TRACKED_IDS = 200_000
# _resolved is the dedup guard and must span the full retention window; the smaller pending cap
# would evict resolved ids early at high throughput and reopen the double-count window.
DEFAULT_MAX_RESOLVED_IDS = 1_000_000

_V = TypeVar("_V")


@dataclass(slots=True)
class RollingPerformance:
    """Joins scored-features to delayed labels by transaction id. A late label waits until it
    matches or ages out of the event-time window; within the window a match counts once."""

    cost_matrix: CostMatrix
    window_size: int = DEFAULT_WINDOW_SIZE
    retention_seconds: float = DEFAULT_RETENTION_SECONDS
    max_tracked_ids: int = DEFAULT_MAX_TRACKED_IDS
    max_resolved_ids: int = DEFAULT_MAX_RESOLVED_IDS
    _pending: OrderedDict[str, tuple[float, bool, float]] = field(init=False)
    _early_labels: OrderedDict[str, tuple[int, float]] = field(init=False)
    _resolved: OrderedDict[str, float] = field(init=False)
    _matched: deque[tuple[float, int, bool]] = field(init=False)
    _high_water: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self._pending = OrderedDict()
        self._early_labels = OrderedDict()
        self._resolved = OrderedDict()
        self._matched = deque(maxlen=self.window_size)

    def observe_score(
        self, transaction_id: str, fraud_score: float, decision: bool, *, event_time: float
    ) -> None:
        self._advance(event_time)
        if self._expired(event_time):
            return
        if transaction_id in self._resolved:
            return
        early_label = self._early_labels.pop(transaction_id, None)
        if early_label is not None:
            self._record(transaction_id, fraud_score, decision, early_label[0], event_time)
            return
        self._pending[transaction_id] = (fraud_score, decision, event_time)
        self._pending.move_to_end(transaction_id)
        _cap(self._pending, self.max_tracked_ids)

    def observe_label(self, transaction_id: str, is_fraud: int, *, event_time: float) -> None:
        self._advance(event_time)
        if self._expired(event_time):
            return
        if transaction_id in self._resolved:
            return
        scored = self._pending.pop(transaction_id, None)
        if scored is None:
            self._early_labels[transaction_id] = (is_fraud, event_time)
            self._early_labels.move_to_end(transaction_id)
            _cap(self._early_labels, self.max_tracked_ids)
            return
        self._record(transaction_id, scored[0], scored[1], is_fraud, event_time)

    def rolling_auprc(self) -> float:
        if not self._matched:
            return math.nan
        scores = [score for score, _, _ in self._matched]
        labels = [label for _, label, _ in self._matched]
        return auprc(labels, scores)

    def business_cost_per_txn(self) -> float:
        """Realized USD cost per transaction from served decisions versus outcomes."""
        if not self._matched:
            return math.nan
        total = 0.0
        for _, is_fraud, decision in self._matched:
            if is_fraud == 1 and not decision:
                total += self.cost_matrix.fn_cost_usd
            elif is_fraud == 0 and decision:
                total += self.cost_matrix.fp_cost_usd
        return total / len(self._matched)

    def flagged_rate(self) -> float:
        if not self._matched:
            return math.nan
        flagged = sum(1 for _, _, decision in self._matched if decision)
        return flagged / len(self._matched)

    @property
    def matched_count(self) -> int:
        return len(self._matched)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def current_event_time(self) -> float:
        return self._high_water

    def _record(
        self,
        transaction_id: str,
        fraud_score: float,
        decision: bool,
        label: int,
        event_time: float,
    ) -> None:
        self._matched.append((fraud_score, label, decision))
        self._resolved[transaction_id] = event_time
        _cap(self._resolved, self.max_resolved_ids)

    def _expired(self, event_time: float) -> bool:
        return event_time < self._high_water - self.retention_seconds

    def _advance(self, event_time: float) -> None:
        """Advance the high-water mark and drop entries past retention."""
        if event_time > self._high_water:
            self._high_water = event_time
        cutoff = self._high_water - self.retention_seconds
        _evict_before(self._pending, cutoff, lambda entry: entry[2])
        _evict_before(self._early_labels, cutoff, lambda entry: entry[1])
        _evict_before(self._resolved, cutoff, lambda entry: entry)


def _evict_before(
    mapping: OrderedDict[str, _V], cutoff: float, event_time_of: Callable[[_V], float]
) -> None:
    """Drop entries older than the cutoff, front-only: out-of-order entries are over-retained,
    never dropped early."""
    while mapping:
        oldest = next(iter(mapping.values()))
        if event_time_of(oldest) >= cutoff:
            break
        mapping.popitem(last=False)


def _cap(mapping: OrderedDict[str, _V], max_size: int) -> None:
    """Drop the oldest entries once the table exceeds its ceiling, so a stalled clock that
    disables retention eviction cannot grow the join tables without bound."""
    while len(mapping) > max_size:
        mapping.popitem(last=False)
