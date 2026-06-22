"""Rolling model performance from a bounded join of scores and delayed labels."""

from __future__ import annotations

import math
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import TypeVar

from fraud.evaluation.business import CostMatrix
from fraud.evaluation.metrics import auprc

DEFAULT_WINDOW_SIZE = 5000
# Must exceed predictions in flight before their labels arrive (arrival_rate * max_lag); the
# holdout replay peaks near 20k. Too small evicts a pending score before its label lands.
DEFAULT_JOIN_RETENTION = 30000

_K = TypeVar("_K")
_V = TypeVar("_V")


@dataclass(slots=True)
class RollingPerformance:
    """Joins scored-features with delayed labels by transaction id to track decay.

    Labels arrive long after their prediction, so unmatched scores wait in a
    bounded buffer until their label shows up or they age out (the join window).
    Matches are idempotent: a transaction is counted once, so at-least-once
    redelivery of either side never double counts the rolling metrics.
    """

    cost_matrix: CostMatrix
    window_size: int = DEFAULT_WINDOW_SIZE
    join_retention: int = DEFAULT_JOIN_RETENTION
    _pending: OrderedDict[str, tuple[float, bool]] = field(init=False)
    _early_labels: OrderedDict[str, int] = field(init=False)
    _resolved: OrderedDict[str, None] = field(init=False)
    _matched: deque[tuple[float, int, bool]] = field(init=False)

    def __post_init__(self) -> None:
        self._pending = OrderedDict()
        self._early_labels = OrderedDict()
        self._resolved = OrderedDict()
        self._matched = deque(maxlen=self.window_size)

    def observe_score(self, transaction_id: str, fraud_score: float, decision: bool) -> None:
        if transaction_id in self._resolved:
            return
        early_label = self._early_labels.pop(transaction_id, None)
        if early_label is not None:
            self._record(transaction_id, fraud_score, decision, early_label)
            return
        self._pending[transaction_id] = (fraud_score, decision)
        self._pending.move_to_end(transaction_id)
        _cap(self._pending, self.join_retention)

    def observe_label(self, transaction_id: str, is_fraud: int) -> None:
        if transaction_id in self._resolved:
            return
        scored = self._pending.pop(transaction_id, None)
        if scored is None:
            self._early_labels[transaction_id] = is_fraud
            self._early_labels.move_to_end(transaction_id)
            _cap(self._early_labels, self.join_retention)
            return
        self._record(transaction_id, scored[0], scored[1], is_fraud)

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

    def _record(self, transaction_id: str, fraud_score: float, decision: bool, label: int) -> None:
        self._matched.append((fraud_score, label, decision))
        self._resolved[transaction_id] = None
        _cap(self._resolved, self.join_retention)


def _cap(mapping: OrderedDict[_K, _V], limit: int) -> None:
    while len(mapping) > limit:
        mapping.popitem(last=False)
