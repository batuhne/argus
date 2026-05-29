import math

import pytest

from fraud.evaluation.gate import (
    GATE_AUPRC_REGRESSION,
    GATE_BOOTSTRAP,
    GATE_BOTH_REGRESSION,
    GATE_CHALLENGER_WINS,
    GATE_COST_REGRESSION,
    GATE_INVALID_CHALLENGER,
    GateMetrics,
    GateTolerances,
    decide,
)


def _challenger(auprc: float = 0.8, cost: float = 0.10) -> GateMetrics:
    return GateMetrics(auprc=auprc, expected_cost_per_tx=cost)


def _champion(auprc: float = 0.7, cost: float = 0.15) -> GateMetrics:
    return GateMetrics(auprc=auprc, expected_cost_per_tx=cost)


def test_gate_bootstrap_promotes_when_no_champion() -> None:
    decision = decide(_challenger(), champion=None)

    assert decision.promote
    assert decision.reason == GATE_BOOTSTRAP


def test_gate_promotes_when_challenger_dominates() -> None:
    decision = decide(_challenger(), _champion())

    assert decision.promote
    assert decision.reason == GATE_CHALLENGER_WINS


def test_gate_rejects_when_only_auprc_regresses() -> None:
    decision = decide(_challenger(auprc=0.6), _champion(auprc=0.7, cost=0.20))

    assert not decision.promote
    assert decision.reason == GATE_AUPRC_REGRESSION


def test_gate_rejects_when_only_cost_regresses() -> None:
    decision = decide(_challenger(auprc=0.9, cost=0.30), _champion(auprc=0.7, cost=0.15))

    assert not decision.promote
    assert decision.reason == GATE_COST_REGRESSION


def test_gate_rejects_when_both_regress() -> None:
    decision = decide(_challenger(auprc=0.5, cost=0.30), _champion(auprc=0.7, cost=0.15))

    assert not decision.promote
    assert decision.reason == GATE_BOTH_REGRESSION


def test_gate_refuses_promotion_when_challenger_metric_is_nan() -> None:
    decision = decide(_challenger(auprc=math.nan), _champion())

    assert not decision.promote
    assert decision.reason == GATE_INVALID_CHALLENGER


def test_gate_raises_when_champion_metric_is_nan() -> None:
    with pytest.raises(ValueError, match="corrupt state"):
        decide(_challenger(), _champion(auprc=math.nan))


def test_gate_treats_equal_metrics_as_pass_at_zero_tolerance() -> None:
    decision = decide(_challenger(auprc=0.7, cost=0.15), _champion(auprc=0.7, cost=0.15))

    assert decision.promote
    assert decision.reason == GATE_CHALLENGER_WINS


def test_gate_tolerances_reject_negative_values() -> None:
    with pytest.raises(ValueError, match="auprc_tolerance"):
        GateTolerances(auprc_tolerance=-0.01)
    with pytest.raises(ValueError, match="cost_tolerance"):
        GateTolerances(cost_tolerance=-0.01)
