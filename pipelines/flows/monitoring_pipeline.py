from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
from prefect import flow, get_run_logger, task

from fraud.common.logging import configure_logging
from fraud.config import get_settings
from fraud.monitoring.baseline import load_baseline
from fraud.monitoring.config import MonitoringConfig
from fraud.monitoring.drift import build_drift_report, compute_feature_drift
from fraud.training.dataset import SPLITS, build_training_frame
from fraud.training.features import FEATURE_COLUMNS, build_xy

MONITORING_EXPERIMENT = "argus-monitoring"


def validated_split(current_split: str) -> str:
    """Guard the split name so it cannot escape the known dataset files."""
    if current_split not in SPLITS:
        raise ValueError(f"current_split must be one of {SPLITS}, got {current_split!r}")
    return current_split


@task
def evaluate_drift(cfg: MonitoringConfig, current_split: str) -> dict[str, float]:
    reference = load_baseline(cfg)
    current, _ = build_xy(build_training_frame(validated_split(current_split)))
    snapshot = build_drift_report(reference, current, FEATURE_COLUMNS)
    drift = compute_feature_drift(
        reference, current, FEATURE_COLUMNS, psi_threshold=cfg.psi_threshold
    )
    _log_to_mlflow(cfg, snapshot, drift.psi, drift.drifted_features, current_split)
    return drift.psi


@flow(name="argus-monitoring")
def monitoring_flow(current_split: str = "test") -> dict[str, float]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    cfg = MonitoringConfig.from_settings()
    current_split = validated_split(current_split)
    logger = get_run_logger()
    logger.info("drift evaluation: split=%s baseline_alias=%s", current_split, cfg.champion_alias)
    psi = evaluate_drift(cfg, current_split)
    logger.info("drift evaluation done: max_psi=%.4f", max(psi.values(), default=0.0))
    return psi


def _log_to_mlflow(
    cfg: MonitoringConfig,
    snapshot: object,
    psi: dict[str, float],
    drifted_features: list[str],
    current_split: str,
) -> None:
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(MONITORING_EXPERIMENT)
    with mlflow.start_run(run_name=f"drift_{current_split}"):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "drift_report.html"
            snapshot.save_html(str(report_path))  # type: ignore[attr-defined]
            mlflow.log_artifact(str(report_path))
        for feature, value in psi.items():
            mlflow.log_metric(f"psi_{feature}", value)
        mlflow.log_metric("drifted_features", len(drifted_features))
        mlflow.set_tag("current_split", current_split)
        mlflow.set_tag("psi_threshold", str(cfg.psi_threshold))


if __name__ == "__main__":
    monitoring_flow()
