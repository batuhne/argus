from __future__ import annotations

from prefect import flow, get_run_logger, task

from fraud.common.logging import configure_logging
from fraud.config import get_settings
from fraud.training.train import TrainingConfig, TrainingResult, run_training


@task
def run_training_task(cfg: TrainingConfig) -> TrainingResult:
    return run_training(cfg)


@flow(name="argus-training")
def training_flow(n_trials: int | None = None, timeout: int | None = None) -> str:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    cfg = TrainingConfig.from_settings(n_trials=n_trials, timeout=timeout)
    logger = get_run_logger()
    logger.info(
        "starting training: n_trials=%s timeout=%s",
        cfg.optuna_n_trials,
        cfg.optuna_timeout,
    )
    result = run_training_task(cfg)
    logger.info(
        "training done: run_id=%s version=%d primary=%s val_auprc=%.4f",
        result.run_id,
        result.model_version,
        result.primary.family,
        result.primary.val_metrics["auprc"],
    )
    return result.run_id


if __name__ == "__main__":
    training_flow()
