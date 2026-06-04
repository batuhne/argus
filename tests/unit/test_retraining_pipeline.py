from pipelines.flows.retraining_pipeline import (
    RETRAIN_REASON_DRIFT,
    RETRAIN_REASON_SCHEDULED,
    outcome_from_result,
)

from fraud.evaluation.gate import (
    GATE_AUPRC_REGRESSION,
    GATE_BOOTSTRAP,
    GateDecision,
    GateMetrics,
)
from fraud.evaluation.threshold import ThresholdDecision
from fraud.training.train import GateOutcome, ModelResult, TrainingResult


def _result(*, promote: bool, reason: str) -> TrainingResult:
    challenger = GateMetrics(auprc=0.1, expected_cost_per_tx=2.0)
    decision = GateDecision(promote=promote, reason=reason, challenger=challenger, champion=None)
    threshold = ThresholdDecision(
        threshold=0.5,
        expected_cost=10.0,
        expected_cost_per_tx=2.0,
        recall=0.5,
        precision=0.5,
        flagged_rate=0.1,
    )
    gate = GateOutcome(
        threshold=threshold,
        test_expected_cost_total=10.0,
        test_expected_cost_per_tx=2.0,
        decision=decision,
    )
    primary = ModelResult(
        family="xgboost",
        model=None,
        train_metrics={},
        val_metrics={"auprc": 0.1},
        test_metrics={"auprc": 0.1},
    )
    return TrainingResult(run_id="run-1", model_version=7, primary=primary, gate=gate)


def test_outcome_maps_promoted_challenger() -> None:
    outcome = outcome_from_result(
        _result(promote=True, reason=GATE_BOOTSTRAP), RETRAIN_REASON_SCHEDULED
    )
    assert outcome.promoted
    assert outcome.run_id == "run-1"
    assert outcome.model_version == 7
    assert outcome.reason == RETRAIN_REASON_SCHEDULED
    assert outcome.gate_reason == GATE_BOOTSTRAP


def test_outcome_maps_held_back_challenger() -> None:
    outcome = outcome_from_result(
        _result(promote=False, reason=GATE_AUPRC_REGRESSION), RETRAIN_REASON_DRIFT
    )
    assert not outcome.promoted
    assert outcome.reason == RETRAIN_REASON_DRIFT
    assert outcome.gate_reason == GATE_AUPRC_REGRESSION
