from collections.abc import Callable, Iterator

import pytest

from fraud.serving.canary import (
    CANARY_AGREEMENT_BREACH,
    CANARY_AUPRC_REGRESSION,
    CANARY_AWAITING_QUALITY_SIGNAL,
    CANARY_ERROR_BREACH,
    CANARY_HOLD_EXHAUSTED,
    CANARY_INSUFFICIENT_SAMPLES,
    CANARY_LATENCY_BREACH,
    CANARY_OK,
    CanaryAction,
    CanaryGates,
    CanaryObservation,
    CanaryRamp,
    decide_step,
    run_canary,
)

GATES = CanaryGates(
    max_p99_latency_ms=50.0,
    max_error_rate=0.001,
    min_auprc_ratio=0.98,
    min_agreement=0.95,
)
MIN_SAMPLES = 500


def _healthy(**overrides: float | int | None) -> CanaryObservation:
    base: dict[str, float | int | None] = {
        "p99_latency_ms": 20.0,
        "error_rate": 0.0,
        "agreement": 0.99,
        "champion_auprc": 0.1,
        "samples": 1000,
        "canary_auprc": 0.1,
    }
    base.update(overrides)
    return CanaryObservation(**base)  # type: ignore[arg-type]


def test_decide_step_promotes_when_all_gates_pass() -> None:
    decision = decide_step(GATES, _healthy(), MIN_SAMPLES)
    assert decision.action is CanaryAction.PROMOTE
    assert decision.reason == CANARY_OK


def test_decide_step_promotes_on_online_signal_before_labels() -> None:
    decision = decide_step(GATES, _healthy(canary_auprc=None), MIN_SAMPLES)
    assert decision.action is CanaryAction.PROMOTE


def test_decide_step_rolls_back_on_latency_breach() -> None:
    decision = decide_step(GATES, _healthy(p99_latency_ms=80.0), MIN_SAMPLES)
    assert decision.action is CanaryAction.ROLLBACK
    assert CANARY_LATENCY_BREACH in decision.breaches


def test_decide_step_rolls_back_on_error_breach() -> None:
    decision = decide_step(GATES, _healthy(error_rate=0.02), MIN_SAMPLES)
    assert decision.action is CanaryAction.ROLLBACK
    assert CANARY_ERROR_BREACH in decision.breaches


def test_slo_breach_outranks_insufficient_samples() -> None:
    decision = decide_step(GATES, _healthy(p99_latency_ms=80.0, samples=1), MIN_SAMPLES)
    assert decision.action is CanaryAction.ROLLBACK
    assert decision.reason == CANARY_LATENCY_BREACH


def test_decide_step_holds_until_warm() -> None:
    decision = decide_step(GATES, _healthy(samples=10), MIN_SAMPLES)
    assert decision.action is CanaryAction.HOLD
    assert decision.reason == CANARY_INSUFFICIENT_SAMPLES


def test_decide_step_rolls_back_on_decision_divergence() -> None:
    decision = decide_step(GATES, _healthy(agreement=0.80), MIN_SAMPLES)
    assert decision.action is CanaryAction.ROLLBACK
    assert CANARY_AGREEMENT_BREACH in decision.breaches


def test_decide_step_rolls_back_on_auprc_regression() -> None:
    decision = decide_step(GATES, _healthy(champion_auprc=0.2, canary_auprc=0.1), MIN_SAMPLES)
    assert decision.action is CanaryAction.ROLLBACK
    assert CANARY_AUPRC_REGRESSION in decision.breaches


def test_decide_step_holds_full_promotion_until_labeled_auprc() -> None:
    decision = decide_step(
        GATES, _healthy(canary_auprc=None), MIN_SAMPLES, require_quality_signal=True
    )
    assert decision.action is CanaryAction.HOLD
    assert decision.reason == CANARY_AWAITING_QUALITY_SIGNAL


def test_decide_step_promotes_full_traffic_once_labeled_auprc_clears_gate() -> None:
    decision = decide_step(
        GATES, _healthy(canary_auprc=0.1), MIN_SAMPLES, require_quality_signal=True
    )
    assert decision.action is CanaryAction.PROMOTE


def test_decide_step_full_promotion_still_rolls_back_on_auprc_regression() -> None:
    decision = decide_step(
        GATES,
        _healthy(champion_auprc=0.2, canary_auprc=0.1),
        MIN_SAMPLES,
        require_quality_signal=True,
    )
    assert decision.action is CanaryAction.ROLLBACK
    assert CANARY_AUPRC_REGRESSION in decision.breaches


@pytest.mark.parametrize("steps", [(), (0.5, 0.25), (0.1, 0.1), (0.0, 1.0), (0.5, 1.5)])
def test_canary_ramp_rejects_invalid_steps(steps: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        CanaryRamp(steps)


def test_canary_ramp_accessors() -> None:
    ramp = CanaryRamp((0.05, 0.25, 1.0))
    assert ramp.weight(0) == 0.05
    assert ramp.is_final(2)
    assert not ramp.is_final(0)


def _recorder() -> tuple[list[float], list[str], list[bool]]:
    return [], [], []


def test_run_canary_promotes_through_full_ramp() -> None:
    weights, rollbacks, promotes = _recorder()
    outcome = run_canary(
        CanaryRamp((0.05, 0.25, 1.0)),
        GATES,
        observe=lambda _i, _w: _healthy(),
        apply_weight=weights.append,
        on_promote=lambda: promotes.append(True),
        on_rollback=rollbacks.append,
        min_samples=MIN_SAMPLES,
        max_holds_per_step=3,
    )
    assert outcome.promoted
    assert outcome.final_step == 2
    assert weights == [0.05, 0.25, 1.0]
    assert promotes == [True]
    assert rollbacks == []


def test_run_canary_rolls_back_mid_ramp() -> None:
    weights, rollbacks, promotes = _recorder()

    def observe(step_index: int, _weight: float) -> CanaryObservation:
        return _healthy(p99_latency_ms=90.0) if step_index == 1 else _healthy()

    outcome = run_canary(
        CanaryRamp((0.05, 0.25, 1.0)),
        GATES,
        observe=observe,
        apply_weight=weights.append,
        on_promote=lambda: promotes.append(True),
        on_rollback=rollbacks.append,
        min_samples=MIN_SAMPLES,
        max_holds_per_step=3,
    )
    assert not outcome.promoted
    assert outcome.final_step == 1
    assert outcome.reason == CANARY_LATENCY_BREACH
    assert weights[-1] == 0.0
    assert rollbacks == [CANARY_LATENCY_BREACH]
    assert promotes == []


def _step_sequence(
    observations: list[CanaryObservation],
) -> Callable[[int, float], CanaryObservation]:
    stream: Iterator[CanaryObservation] = iter(observations)
    return lambda _i, _w: next(stream)


def test_run_canary_holds_then_promotes() -> None:
    weights, rollbacks, promotes = _recorder()
    observe = _step_sequence([_healthy(samples=10), _healthy(), _healthy()])
    outcome = run_canary(
        CanaryRamp((0.5, 1.0)),
        GATES,
        observe=observe,
        apply_weight=weights.append,
        on_promote=lambda: promotes.append(True),
        on_rollback=rollbacks.append,
        min_samples=MIN_SAMPLES,
        max_holds_per_step=3,
    )
    assert outcome.promoted
    assert promotes == [True]


def test_run_canary_rolls_back_when_holds_exhausted() -> None:
    weights, rollbacks, promotes = _recorder()
    outcome = run_canary(
        CanaryRamp((0.5, 1.0)),
        GATES,
        observe=lambda _i, _w: _healthy(samples=1),
        apply_weight=weights.append,
        on_promote=lambda: promotes.append(True),
        on_rollback=rollbacks.append,
        min_samples=MIN_SAMPLES,
        max_holds_per_step=2,
    )
    assert not outcome.promoted
    assert outcome.reason == CANARY_HOLD_EXHAUSTED
    assert rollbacks == [CANARY_HOLD_EXHAUSTED]
    assert weights[-1] == 0.0


def test_run_canary_holds_final_step_until_labels_then_promotes() -> None:
    weights, rollbacks, promotes = _recorder()
    # 5% step ramps on the online signal, but full traffic waits for a labeled AUPRC.
    observe = _step_sequence([_healthy(canary_auprc=None), _healthy(canary_auprc=None), _healthy()])
    outcome = run_canary(
        CanaryRamp((0.05, 1.0)),
        GATES,
        observe=observe,
        apply_weight=weights.append,
        on_promote=lambda: promotes.append(True),
        on_rollback=rollbacks.append,
        min_samples=MIN_SAMPLES,
        max_holds_per_step=3,
    )
    assert outcome.promoted
    assert weights == [0.05, 1.0, 1.0]


def test_run_canary_rolls_back_when_full_traffic_never_gets_labels() -> None:
    weights, rollbacks, promotes = _recorder()
    outcome = run_canary(
        CanaryRamp((0.05, 1.0)),
        GATES,
        observe=lambda _i, _w: _healthy(canary_auprc=None),
        apply_weight=weights.append,
        on_promote=lambda: promotes.append(True),
        on_rollback=rollbacks.append,
        min_samples=MIN_SAMPLES,
        max_holds_per_step=2,
    )
    assert not outcome.promoted
    assert outcome.reason == CANARY_HOLD_EXHAUSTED
    assert promotes == []
    assert weights[-1] == 0.0
