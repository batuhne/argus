"""Score the champion on the untouched holdout split, read-only w.r.t. the registry.

Holdout is the latest window, never seen by tuning/threshold/gate. Kept out of the package
__init__ so the metric imports stay free of mlflow and the serving loader.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fraud.common.logging import configure_logging, get_logger
from fraud.config import get_settings
from fraud.dataset import build_eval_frame
from fraud.evaluation.business import CostMatrix, expected_cost
from fraud.evaluation.calibration import brier_score, reliability_curve_figure
from fraud.evaluation.metrics import (
    auprc,
    classification_at_threshold,
    pr_curve_figure,
    recall_at_k,
)
from fraud.model_loader import ChampionLoadConfig, ModelBundle, load_champion
from fraud.params import load_params
from fraud.paths import FEATURE_REPO_DIR, PROCESSED_DIR
from fraud.transforms.features import build_xy

HOLDOUT_SPLIT = "holdout"
log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    tracking_uri: str
    experiment_name: str
    cost_matrix: CostMatrix
    recall_at_k_levels: tuple[float, ...]
    repo_dir: Path
    processed_dir: Path
    artifacts_dir: Path

    @classmethod
    def from_settings(cls) -> BacktestConfig:
        settings = get_settings()
        evaluation = load_params().evaluation
        cost = evaluation.cost_matrix
        return cls(
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow_experiment_name,
            cost_matrix=CostMatrix(fn_cost_usd=cost.fn_cost_usd, fp_cost_usd=cost.fp_cost_usd),
            recall_at_k_levels=evaluation.recall_at_k_levels,
            repo_dir=FEATURE_REPO_DIR,
            processed_dir=PROCESSED_DIR,
            artifacts_dir=Path("artifacts"),
        )


@dataclass(frozen=True, slots=True)
class BacktestReport:
    rows: int
    positives: int
    auprc: float
    recall_at_k: dict[float, float]
    threshold: float
    expected_cost_total_usd: float
    expected_cost_per_tx_usd: float
    precision: float
    recall: float
    flagged_rate: float
    brier: float


def evaluate_holdout(
    y_true: pd.Series,
    y_score: NDArray[np.float64],
    *,
    threshold: float,
    cost_matrix: CostMatrix,
    recall_levels: tuple[float, ...],
) -> BacktestReport:
    """Assemble the report at the decision threshold; y_score must be calibrated probabilities."""
    classification = classification_at_threshold(y_true, y_score, threshold)
    cost_total = expected_cost(y_true, y_score, threshold, cost_matrix)
    rows = len(y_true)
    return BacktestReport(
        rows=rows,
        positives=int(y_true.sum()),
        auprc=auprc(y_true, y_score),
        recall_at_k={k: recall_at_k(y_true, y_score, k) for k in recall_levels},
        threshold=threshold,
        expected_cost_total_usd=cost_total,
        expected_cost_per_tx_usd=float(cost_total / rows) if rows else math.nan,
        precision=classification["precision"],
        recall=classification["recall"],
        flagged_rate=classification["flagged_rate"],
        brier=brier_score(y_true, y_score),
    )


def run_backtest(cfg: BacktestConfig) -> BacktestReport:
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)
    bundle = load_champion(ChampionLoadConfig.from_settings())
    _verify_recorded_champion(bundle.version, cfg.artifacts_dir / "last_run.json")
    x, y = _load_holdout_xy(bundle, cfg)
    calibrated = _calibrated_scores(bundle, x)
    report = evaluate_holdout(
        y,
        calibrated,
        threshold=bundle.threshold,
        cost_matrix=cfg.cost_matrix,
        recall_levels=cfg.recall_at_k_levels,
    )
    _log_backtest_run(report, bundle, y, calibrated)
    _write_marker(cfg.artifacts_dir, report, bundle)
    log.info(
        "backtest_done",
        champion_version=bundle.version,
        holdout_rows=report.rows,
        holdout_auprc=report.auprc,
        holdout_cost_per_tx=report.expected_cost_per_tx_usd,
    )
    return report


def _verify_recorded_champion(loaded_version: int, marker_path: Path) -> None:
    """Raise if the loaded champion is not the version train recorded as champion."""
    if not marker_path.exists():
        log.warning(
            "champion_provenance_check_skipped", reason="marker_absent", path=str(marker_path)
        )
        return
    recorded = json.loads(marker_path.read_text()).get("champion_version")
    if recorded is None:
        log.warning("champion_provenance_check_skipped", reason="champion_version_unrecorded")
        return
    if loaded_version != recorded:
        raise RuntimeError(
            f"backtest loaded champion version {loaded_version}, but {marker_path.name} recorded "
            f"version {recorded}; the champion alias moved between train and backtest"
        )


def _load_holdout_xy(bundle: ModelBundle, cfg: BacktestConfig) -> tuple[pd.DataFrame, pd.Series]:
    frame = build_eval_frame(HOLDOUT_SPLIT, bundle.encoder, cfg.repo_dir, cfg.processed_dir)
    return build_xy(frame)


def _calibrated_scores(bundle: ModelBundle, x: pd.DataFrame) -> NDArray[np.float64]:
    raw = bundle.model.predict_proba(x)[:, 1]
    return bundle.calibrator.predict(raw)


def _log_backtest_run(
    report: BacktestReport, bundle: ModelBundle, y_true: pd.Series, y_score: NDArray[np.float64]
) -> None:
    with mlflow.start_run(run_name="argus_backtest"):
        mlflow.set_tag("run_type", "backtest")
        mlflow.set_tag("champion_version", str(bundle.version))
        mlflow.set_tag("champion_family", bundle.family)
        mlflow.log_metrics(
            {
                "holdout_rows": float(report.rows),
                "holdout_positives": float(report.positives),
                "holdout_auprc": report.auprc,
                "holdout_brier": report.brier,
                "holdout_threshold": report.threshold,
                "holdout_precision": report.precision,
                "holdout_recall": report.recall,
                "holdout_flagged_rate": report.flagged_rate,
                "holdout_expected_cost_total_usd": report.expected_cost_total_usd,
                "holdout_expected_cost_per_tx_usd": report.expected_cost_per_tx_usd,
            }
        )
        for k, value in report.recall_at_k.items():
            mlflow.log_metric(f"holdout_recall_at_{k:.3f}", value)
        mlflow.log_figure(
            pr_curve_figure(y_true, y_score, title="Holdout PR curve"), "pr_curve_holdout.png"
        )
        mlflow.log_figure(
            reliability_curve_figure(y_true, y_score, title="Holdout reliability (calibrated)"),
            "reliability_holdout.png",
        )


def _write_marker(artifacts_dir: Path, report: BacktestReport, bundle: ModelBundle) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "champion_version": bundle.version,
        "champion_family": bundle.family,
        "holdout_rows": report.rows,
        "holdout_positives": report.positives,
        "auprc": report.auprc,
        "recall_at_k": {f"{k:.3f}": value for k, value in report.recall_at_k.items()},
        "threshold": report.threshold,
        "expected_cost": {
            "total_usd": report.expected_cost_total_usd,
            "per_tx_usd": report.expected_cost_per_tx_usd,
        },
        "classification": {
            "precision": report.precision,
            "recall": report.recall,
            "flagged_rate": report.flagged_rate,
        },
        "brier": report.brier,
    }
    tmp = artifacts_dir / "backtest.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(artifacts_dir / "backtest.json")


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    cfg = BacktestConfig.from_settings()
    log.info("backtest_start", tracking_uri=cfg.tracking_uri, experiment=cfg.experiment_name)
    report = run_backtest(cfg)
    print(
        f"holdout AUPRC {report.auprc:.4f} on {report.rows} rows ({report.positives} fraud)",
        flush=True,
    )


if __name__ == "__main__":
    main()
