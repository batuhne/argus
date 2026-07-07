"""Load the frozen training feature baseline logged with the champion model."""

from __future__ import annotations

import random
import time

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException

from fraud.common.logging import get_logger
from fraud.monitoring.config import MonitoringConfig
from fraud.registry import (
    BASELINE_ARTIFACT_PATH,
    get_alias_version,
    get_version_run_id,
)

log = get_logger(__name__)

_RETRY_BASE_SECONDS = 1.0
_RETRY_CAP_SECONDS = 8.0


class BaselineUnavailableError(RuntimeError):
    """No champion baseline in the registry yet; a cold start may be racing training."""


def load_baseline(cfg: MonitoringConfig) -> pd.DataFrame:
    """Download the champion's drift baseline, retrying a cold registry until the deadline."""
    mlflow.set_tracking_uri(cfg.tracking_uri)
    deadline = time.monotonic() + cfg.baseline_load_deadline_seconds
    attempt = 0
    while True:
        try:
            return _load_baseline(cfg)
        except (BaselineUnavailableError, MlflowException, OSError) as exc:
            if time.monotonic() >= deadline:
                log.error("baseline_load_failed", attempts=attempt + 1, error=str(exc))
                raise
            delay = _backoff_delay(attempt)
            log.warning("baseline_load_retry", attempt=attempt, error=str(exc), retry_in=delay)
            time.sleep(delay)
            attempt += 1


def _backoff_delay(attempt: int) -> float:
    capped = min(_RETRY_CAP_SECONDS, _RETRY_BASE_SECONDS * 2.0**attempt)
    return capped + random.uniform(0.0, _RETRY_BASE_SECONDS)


def _load_baseline(cfg: MonitoringConfig) -> pd.DataFrame:
    version = get_alias_version(cfg.model_name, cfg.champion_alias)
    if version is None:
        raise BaselineUnavailableError(
            f"no '{cfg.champion_alias}' alias on model '{cfg.model_name}'; train and promote first"
        )
    run_id = get_version_run_id(cfg.model_name, version)
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=BASELINE_ARTIFACT_PATH
    )
    return pd.read_parquet(local_path)
