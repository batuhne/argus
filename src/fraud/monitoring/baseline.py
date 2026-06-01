"""Load the frozen training feature baseline logged with the champion model."""

from __future__ import annotations

import mlflow
import pandas as pd

from fraud.monitoring.config import MonitoringConfig
from fraud.training.registry import (
    BASELINE_ARTIFACT_PATH,
    get_alias_version,
    get_version_run_id,
)


def load_baseline(cfg: MonitoringConfig) -> pd.DataFrame:
    """Download the champion's reference feature sample for drift comparison."""
    mlflow.set_tracking_uri(cfg.tracking_uri)
    version = get_alias_version(cfg.model_name, cfg.champion_alias)
    if version is None:
        raise RuntimeError(
            f"no '{cfg.champion_alias}' alias on model '{cfg.model_name}'; train and promote first"
        )
    run_id = get_version_run_id(cfg.model_name, version)
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=BASELINE_ARTIFACT_PATH
    )
    return pd.read_parquet(local_path)
