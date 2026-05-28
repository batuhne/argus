from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import pandas as pd
import yaml

from fraud.common.lineage import collect_lineage
from fraud.common.logging import configure_logging, get_logger
from fraud.common.seed import set_seed
from fraud.config import get_settings
from fraud.evaluation.metrics import auprc, pr_curve_figure, recall_at_k
from fraud.evaluation.reports import feature_schema_payload, shap_summary_figure
from fraud.paths import FEATURE_REPO_DIR, PROCESSED_DIR
from fraud.training.dataset import FEATURE_SERVICE, load_splits
from fraud.training.features import build_xy
from fraud.training.models import (
    BoostingHyperparams,
    build_lgb,
    build_xgb,
    compute_scale_pos_weight,
)
from fraud.training.registry import register_candidate
from fraud.training.tune import tune_xgb

Splits = dict[str, tuple[pd.DataFrame, pd.Series]]
log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int
    tracking_uri: str
    experiment_name: str
    model_name: str
    candidate_alias: str
    optuna_n_trials: int
    optuna_timeout: int | None
    shap_sample_size: int
    recall_at_k_levels: tuple[float, ...]
    run_name: str
    repo_dir: Path
    processed_dir: Path
    artifacts_dir: Path

    @classmethod
    def from_settings(
        cls, *, n_trials: int | None = None, timeout: int | None = None
    ) -> TrainingConfig:
        settings = get_settings()
        params = _load_training_params()
        optuna_cfg = params.get("optuna") or {}
        shap_cfg = params.get("shap") or {}
        return cls(
            seed=settings.seed,
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow_experiment_name,
            model_name=settings.argus_model_name,
            candidate_alias=str(params.get("candidate_alias", "candidate")),
            optuna_n_trials=int(
                n_trials if n_trials is not None else optuna_cfg.get("n_trials", 30)
            ),
            optuna_timeout=(
                timeout if timeout is not None else optuna_cfg.get("timeout_seconds", 1800)
            ),
            shap_sample_size=int(shap_cfg.get("sample_size", 2000)),
            recall_at_k_levels=tuple(
                float(k) for k in params.get("recall_at_k_levels", [0.005, 0.01, 0.05])
            ),
            run_name="argus_training",
            repo_dir=FEATURE_REPO_DIR,
            processed_dir=PROCESSED_DIR,
            artifacts_dir=Path("artifacts"),
        )


@dataclass(frozen=True, slots=True)
class ModelResult:
    family: str
    model: Any
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    run_id: str
    model_version: int
    primary: ModelResult


def run_training(cfg: TrainingConfig) -> TrainingResult:
    splits = _load_and_split(cfg)
    return train_with_splits(cfg, splits)


def train_with_splits(cfg: TrainingConfig, splits: Splits) -> TrainingResult:
    _seed_everything(cfg.seed)
    _configure_mlflow(cfg)
    with mlflow.start_run(run_name=cfg.run_name) as parent:
        _log_lineage_tags(splits)
        best = _sweep_xgb(splits, cfg)
        primary = _train_and_pick_primary(best, splits, cfg)
        _log_artifacts(primary, splits, cfg)
        version = register_candidate(
            parent.info.run_id, cfg.model_name, alias=cfg.candidate_alias
        )
        mlflow.set_tag("model_version", str(version))
    return TrainingResult(run_id=parent.info.run_id, model_version=version, primary=primary)


def _seed_everything(seed: int) -> None:
    set_seed(seed)


def _configure_mlflow(cfg: TrainingConfig) -> None:
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)


def _load_and_split(cfg: TrainingConfig) -> Splits:
    frames = load_splits(cfg.repo_dir, cfg.processed_dir)
    return {split: build_xy(frame) for split, frame in frames.items()}


def _log_lineage_tags(splits: Splits) -> None:
    tags = collect_lineage().to_mlflow_tags()
    tags["feature_service"] = FEATURE_SERVICE
    for split, (_, y) in splits.items():
        tags[f"split_{split}_rows"] = str(len(y))
        tags[f"split_{split}_positives"] = str(int(y.sum()))
    mlflow.set_tags(tags)


def _sweep_xgb(splits: Splits, cfg: TrainingConfig) -> BoostingHyperparams:
    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]
    with mlflow.start_run(run_name="optuna_sweep", nested=True):
        best, best_value = tune_xgb(
            x_train,
            y_train,
            x_val,
            y_val,
            n_trials=cfg.optuna_n_trials,
            timeout=cfg.optuna_timeout,
            seed=cfg.seed,
        )
        mlflow.log_params(
            {
                "sweep_n_trials": cfg.optuna_n_trials,
                "sweep_timeout": cfg.optuna_timeout or 0,
            }
        )
        mlflow.log_metric("val_auprc_sweep_best", best_value)
    return best


def _train_and_pick_primary(
    best: BoostingHyperparams, splits: Splits, cfg: TrainingConfig
) -> ModelResult:
    _, y_train = splits["train"]
    scale_pos_weight = compute_scale_pos_weight(y_train)
    xgb_result = _fit_and_evaluate_xgb(best, scale_pos_weight, splits, cfg)
    lgb_result = _fit_and_evaluate_lgb(best, scale_pos_weight, splits, cfg)
    _log_family_metrics(xgb_result)
    _log_family_metrics(lgb_result)
    mlflow.log_params({f"best_{key}": value for key, value in asdict(best).items()})
    primary = (
        xgb_result
        if xgb_result.val_metrics["auprc"] >= lgb_result.val_metrics["auprc"]
        else lgb_result
    )
    mlflow.set_tag("primary_family", primary.family)
    return primary


def _fit_and_evaluate_xgb(
    best: BoostingHyperparams, scale_pos_weight: float, splits: Splits, cfg: TrainingConfig
) -> ModelResult:
    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]
    model = build_xgb(best, scale_pos_weight=scale_pos_weight, seed=cfg.seed)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    metrics = _evaluate_on_splits(model, splits, cfg.recall_at_k_levels)
    return ModelResult("xgboost", model, *metrics)


def _fit_and_evaluate_lgb(
    best: BoostingHyperparams, scale_pos_weight: float, splits: Splits, cfg: TrainingConfig
) -> ModelResult:
    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]
    model = build_lgb(best, scale_pos_weight=scale_pos_weight, seed=cfg.seed)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(best.early_stopping_rounds, verbose=False)],
    )
    metrics = _evaluate_on_splits(model, splits, cfg.recall_at_k_levels)
    return ModelResult("lightgbm", model, *metrics)


def _evaluate_on_splits(
    model: Any, splits: Splits, recall_levels: tuple[float, ...]
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    def metrics_for(x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        scores = model.predict_proba(x)[:, 1]
        result = {"auprc": auprc(y, scores)}
        for k in recall_levels:
            result[f"recall_at_{k:.3f}"] = recall_at_k(y, scores, k)
        return result

    return (
        metrics_for(*splits["train"]),
        metrics_for(*splits["val"]),
        metrics_for(*splits["test"]),
    )


def _log_family_metrics(result: ModelResult) -> None:
    for split, metrics in (
        ("train", result.train_metrics),
        ("val", result.val_metrics),
        ("test", result.test_metrics),
    ):
        for name, value in metrics.items():
            mlflow.log_metric(f"{result.family}_{split}_{name}", value)


def _log_artifacts(primary: ModelResult, splits: Splits, cfg: TrainingConfig) -> None:
    x_train, _ = splits["train"]
    x_val, y_val = splits["val"]

    scores = primary.model.predict_proba(x_val)[:, 1]
    pr_figure = pr_curve_figure(y_val, scores, title=f"{primary.family} PR curve (val)")
    mlflow.log_figure(pr_figure, "pr_curve_val.png")

    sample_size = min(cfg.shap_sample_size, len(x_train))
    sample = x_train.sample(sample_size, random_state=cfg.seed)
    shap_figure = shap_summary_figure(primary.model, sample)
    mlflow.log_figure(shap_figure, "shap_summary_train.png")

    mlflow.log_dict(feature_schema_payload(x_train), "feature_schema.json")
    _log_model(primary, input_example=x_train.head(5))


def _log_model(primary: ModelResult, *, input_example: pd.DataFrame) -> None:
    if primary.family == "xgboost":
        mlflow.xgboost.log_model(primary.model, name="model", input_example=input_example)
    else:
        mlflow.lightgbm.log_model(primary.model, name="model", input_example=input_example)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    cfg = TrainingConfig.from_settings()
    result = run_training(cfg)
    _write_run_marker(cfg.artifacts_dir, result)
    log.info(
        "training_done",
        run_id=result.run_id,
        model_version=result.model_version,
        primary=result.primary.family,
        val_auprc=result.primary.val_metrics["auprc"],
    )


def _write_run_marker(artifacts_dir: Path, result: TrainingResult) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": result.run_id,
        "model_version": result.model_version,
        "primary": result.primary.family,
        "metrics": result.primary.val_metrics,
    }
    (artifacts_dir / "last_run.json").write_text(json.dumps(payload, indent=2))


def _load_training_params(path: Path = Path("params.yaml")) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        return {}
    section = data.get("training") or {}
    return section if isinstance(section, dict) else {}


if __name__ == "__main__":
    main()
