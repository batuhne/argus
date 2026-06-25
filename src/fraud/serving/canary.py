"""Canary ramp controller: gate each traffic step on SLO and quality, else roll back."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

CANARY_OK = "ok"
CANARY_INSUFFICIENT_SAMPLES = "insufficient_samples"
CANARY_LATENCY_BREACH = "latency_breach"
CANARY_ERROR_BREACH = "error_rate_breach"
CANARY_AGREEMENT_BREACH = "agreement_breach"
CANARY_AUPRC_REGRESSION = "auprc_regression"
CANARY_AWAITING_QUALITY_SIGNAL = "awaiting_quality_signal"
CANARY_HOLD_EXHAUSTED = "hold_exhausted"


class CanaryAction(StrEnum):
    PROMOTE = "promote"
    HOLD = "hold"
    ROLLBACK = "rollback"


@dataclass(frozen=True, slots=True)
class CanaryGates:
    max_p99_latency_ms: float
    max_error_rate: float
    min_auprc_ratio: float
    min_agreement: float


@dataclass(frozen=True, slots=True)
class CanaryObservation:
    p99_latency_ms: float
    error_rate: float
    agreement: float
    champion_auprc: float
    samples: int
    canary_auprc: float | None = None


@dataclass(frozen=True, slots=True)
class StepDecision:
    action: CanaryAction
    reason: str
    breaches: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CanaryRamp:
    steps: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("canary ramp needs at least one traffic step")
        if list(self.steps) != sorted(self.steps) or len(set(self.steps)) != len(self.steps):
            raise ValueError(f"canary steps must strictly increase, got {self.steps}")
        if not all(0.0 < weight <= 1.0 for weight in self.steps):
            raise ValueError(f"canary steps must lie in (0, 1], got {self.steps}")

    def weight(self, step_index: int) -> float:
        return self.steps[step_index]

    def is_final(self, step_index: int) -> bool:
        return step_index == len(self.steps) - 1


@dataclass(frozen=True, slots=True)
class CanaryOutcome:
    promoted: bool
    final_step: int
    reason: str
    history: tuple[StepDecision, ...]


def decide_step(
    gates: CanaryGates,
    obs: CanaryObservation,
    min_samples: int,
    *,
    require_quality_signal: bool = False,
) -> StepDecision:
    """A latency or error breach rolls back immediately; full promotion needs a labeled AUPRC."""
    slo_breaches = _slo_breaches(gates, obs)
    if slo_breaches:
        return StepDecision(CanaryAction.ROLLBACK, slo_breaches[0], slo_breaches)
    if obs.samples < min_samples:
        return StepDecision(CanaryAction.HOLD, CANARY_INSUFFICIENT_SAMPLES, ())
    quality_breaches = _quality_breaches(gates, obs)
    if quality_breaches:
        return StepDecision(CanaryAction.ROLLBACK, quality_breaches[0], quality_breaches)
    if require_quality_signal and obs.canary_auprc is None:
        return StepDecision(CanaryAction.HOLD, CANARY_AWAITING_QUALITY_SIGNAL, ())
    return StepDecision(CanaryAction.PROMOTE, CANARY_OK, ())


def _slo_breaches(gates: CanaryGates, obs: CanaryObservation) -> tuple[str, ...]:
    breaches: list[str] = []
    if obs.p99_latency_ms > gates.max_p99_latency_ms:
        breaches.append(CANARY_LATENCY_BREACH)
    if obs.error_rate > gates.max_error_rate:
        breaches.append(CANARY_ERROR_BREACH)
    return tuple(breaches)


def _quality_breaches(gates: CanaryGates, obs: CanaryObservation) -> tuple[str, ...]:
    breaches: list[str] = []
    if obs.agreement < gates.min_agreement:
        breaches.append(CANARY_AGREEMENT_BREACH)
    if obs.canary_auprc is not None and (
        obs.canary_auprc < gates.min_auprc_ratio * obs.champion_auprc
    ):
        breaches.append(CANARY_AUPRC_REGRESSION)
    return tuple(breaches)


@dataclass(slots=True)
class _RampState:
    history: list[StepDecision] = field(default_factory=list)


def run_canary(
    ramp: CanaryRamp,
    gates: CanaryGates,
    *,
    observe: Callable[[int, float], CanaryObservation],
    apply_weight: Callable[[float], None],
    on_promote: Callable[[], None],
    on_rollback: Callable[[str], None],
    min_samples: int,
    max_holds_per_step: int,
) -> CanaryOutcome:
    """Drive the ramp with injected effects so the policy stays unit-testable."""
    state = _RampState()
    step_index = 0
    holds = 0
    while True:
        weight = ramp.weight(step_index)
        apply_weight(weight)
        decision = decide_step(
            gates,
            observe(step_index, weight),
            min_samples,
            require_quality_signal=ramp.is_final(step_index),
        )
        state.history.append(decision)

        if decision.action is CanaryAction.ROLLBACK:
            return _rollback(state, step_index, decision.reason, apply_weight, on_rollback)
        if decision.action is CanaryAction.HOLD:
            holds += 1
            if holds > max_holds_per_step:
                return _rollback(
                    state, step_index, CANARY_HOLD_EXHAUSTED, apply_weight, on_rollback
                )
            continue

        holds = 0
        if ramp.is_final(step_index):
            on_promote()
            return CanaryOutcome(True, step_index, CANARY_OK, tuple(state.history))
        step_index += 1


def _rollback(
    state: _RampState,
    step_index: int,
    reason: str,
    apply_weight: Callable[[float], None],
    on_rollback: Callable[[str], None],
) -> CanaryOutcome:
    apply_weight(0.0)
    on_rollback(reason)
    return CanaryOutcome(False, step_index, reason, tuple(state.history))
