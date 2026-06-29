"""Promotion gate: decide whether a candidate beats the champion on AUPRC and cost."""

from __future__ import annotations

import math
from dataclasses import dataclass

GATE_BOOTSTRAP = "bootstrap"
GATE_CHALLENGER_WINS = "challenger_wins"
GATE_AUPRC_REGRESSION = "auprc_regression"
GATE_COST_REGRESSION = "cost_regression"
GATE_BOTH_REGRESSION = "both_regression"
GATE_INVALID_CHALLENGER = "invalid_challenger"


@dataclass(frozen=True, slots=True)
class GateMetrics:
    auprc: float
    expected_cost_per_tx: float


@dataclass(frozen=True, slots=True)
class GateTolerances:
    auprc_tolerance: float = 0.0
    cost_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.auprc_tolerance < 0.0:
            raise ValueError(f"auprc_tolerance must be non-negative, got {self.auprc_tolerance}")
        if self.cost_tolerance < 0.0:
            raise ValueError(f"cost_tolerance must be non-negative, got {self.cost_tolerance}")


DEFAULT_TOLERANCES = GateTolerances()


@dataclass(frozen=True, slots=True)
class GateDecision:
    promote: bool
    reason: str
    challenger: GateMetrics
    champion: GateMetrics | None


def decide(
    challenger: GateMetrics,
    champion: GateMetrics | None,
    *,
    tolerances: GateTolerances = DEFAULT_TOLERANCES,
) -> GateDecision:
    """Promote only when the challenger beats the champion on AUPRC and expected cost."""
    if _has_nan(challenger):
        return GateDecision(False, GATE_INVALID_CHALLENGER, challenger, champion)
    if champion is None:
        return GateDecision(True, GATE_BOOTSTRAP, challenger, None)
    if _has_nan(champion):
        raise ValueError("champion metrics contain NaN; refusing to gate against corrupt state")

    auprc_pass = challenger.auprc >= champion.auprc - tolerances.auprc_tolerance
    cost_pass = (
        challenger.expected_cost_per_tx <= champion.expected_cost_per_tx + tolerances.cost_tolerance
    )
    return GateDecision(
        promote=auprc_pass and cost_pass,
        reason=_reason_for(auprc_pass, cost_pass),
        challenger=challenger,
        champion=champion,
    )


def _reason_for(auprc_pass: bool, cost_pass: bool) -> str:
    if auprc_pass and cost_pass:
        return GATE_CHALLENGER_WINS
    if not auprc_pass and not cost_pass:
        return GATE_BOTH_REGRESSION
    if not auprc_pass:
        return GATE_AUPRC_REGRESSION
    return GATE_COST_REGRESSION


def _has_nan(metrics: GateMetrics) -> bool:
    return math.isnan(metrics.auprc) or math.isnan(metrics.expected_cost_per_tx)
