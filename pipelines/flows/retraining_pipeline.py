from __future__ import annotations

from dataclasses import dataclass

from prefect import flow, get_run_logger, task

from fraud.common.logging import configure_logging
from fraud.config import get_settings
from fraud.params import load_params
from fraud.training.train import TrainingConfig, TrainingResult, run_training

RETRAIN_REASON_SCHEDULED = "scheduled"
RETRAIN_REASON_DRIFT = "drift_alert"


@dataclass(frozen=True, slots=True)
class RetrainingOutcome:
    run_id: str
    model_version: int
    promoted: bool
    reason: str
    gate_reason: str


def outcome_from_result(result: TrainingResult, reason: str) -> RetrainingOutcome:
    return RetrainingOutcome(
        run_id=result.run_id,
        model_version=result.model_version,
        promoted=result.gate.decision.promote,
        reason=reason,
        gate_reason=result.gate.decision.reason,
    )


@task
def run_training_task(cfg: TrainingConfig) -> TrainingResult:
    return run_training(cfg)


@flow(name="argus-retraining")
def retraining_flow(
    reason: str = RETRAIN_REASON_SCHEDULED,
    n_trials: int | None = None,
    timeout: int | None = None,
) -> RetrainingOutcome:
    """Retrain the challenger and let the promotion gate keep the better model live."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    defaults = load_params().retraining
    cfg = TrainingConfig.from_settings(
        n_trials=n_trials if n_trials is not None else defaults.n_trials,
        timeout=timeout if timeout is not None else defaults.timeout_seconds,
    )
    logger = get_run_logger()
    logger.info("retraining triggered: reason=%s n_trials=%s", reason, cfg.optuna_n_trials)
    outcome = outcome_from_result(run_training_task(cfg), reason)
    logger.info(
        "retraining done: version=%d promoted=%s gate=%s",
        outcome.model_version,
        outcome.promoted,
        outcome.gate_reason,
    )
    return outcome


if __name__ == "__main__":
    retraining_flow()
