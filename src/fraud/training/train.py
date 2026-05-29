from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from fraud.common.lineage import collect_lineage
from fraud.common.logging import configure_logging, get_logger
from fraud.common.seed import set_seed
from fraud.config import get_settings
from fraud.evaluation.business import CostMatrix, expected_cost
from fraud.evaluation.calibration import (
    CalibrationResult,
    fit_isotonic,
    reliability_curve_figure,
)
from fraud.evaluation.gate import (
    GateDecision,
    GateMetrics,
    GateTolerances,
    decide,
)
from fraud.evaluation.metrics import auprc, pr_curve_figure, recall_at_k
from fraud.evaluation.reports import feature_schema_payload, shap_summary_figure
from fraud.evaluation.threshold import (
    ThresholdConstraints,
    ThresholdDecision,
    select_threshold,
)
from fraud.paths import FEATURE_REPO_DIR, PROCESSED_DIR
from fraud.training.dataset import FEATURE_SERVICE, load_splits
from fraud.training.features import build_xy
from fraud.training.models import (
    BoostingHyperparams,
    build_lgb,
    build_xgb,
    compute_scale_pos_weight,
)
from fraud.training.registry import (
    CHAMPION_TAG_AUPRC,
    CHAMPION_TAG_COST_PER_TX,
    attach_alias,
    get_alias_version,
    get_version_tags,
    write_version_tags,
)
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
    champion_alias: str
    optuna_n_trials: int
    optuna_timeout: int | None
    shap_sample_size: int
    recall_at_k_levels: tuple[float, ...]
    cost_matrix: CostMatrix
    threshold_constraints: ThresholdConstraints
    gate_tolerances: GateTolerances
    run_name: str
    repo_dir: Path
    processed_dir: Path
    artifacts_dir: Path

    @classmethod
    def from_settings(
        cls, *, n_trials: int | None = None, timeout: int | None = None
    ) -> TrainingConfig:
        settings = get_settings()
        training_params = _load_section_params("training")
        evaluation_params = _load_section_params("evaluation")
        optuna_cfg = training_params.get("optuna") or {}
        shap_cfg = training_params.get("shap") or {}
        return cls(
            seed=settings.seed,
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow_experiment_name,
            model_name=settings.argus_model_name,
            candidate_alias=str(training_params.get("candidate_alias", "candidate")),
            champion_alias=str(evaluation_params.get("champion_alias", "champion")),
            optuna_n_trials=int(
                n_trials if n_trials is not None else optuna_cfg.get("n_trials", 30)
            ),
            optuna_timeout=(
                timeout if timeout is not None else optuna_cfg.get("timeout_seconds", 1800)
            ),
            shap_sample_size=int(shap_cfg.get("sample_size", 2000)),
            recall_at_k_levels=tuple(
                float(k) for k in training_params.get("recall_at_k_levels", [0.005, 0.01, 0.05])
            ),
            cost_matrix=_cost_matrix_from(evaluation_params),
            threshold_constraints=_threshold_constraints_from(evaluation_params),
            gate_tolerances=_gate_tolerances_from(evaluation_params),
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
class GateOutcome:
    threshold: ThresholdDecision
    test_expected_cost_total: float
    test_expected_cost_per_tx: float
    decision: GateDecision


@dataclass(frozen=True, slots=True)
class TrainingResult:
    run_id: str
    model_version: int
    primary: ModelResult
    gate: GateOutcome


def run_training(cfg: TrainingConfig) -> TrainingResult:
    log.info("loading_splits", repo_dir=str(cfg.repo_dir), processed_dir=str(cfg.processed_dir))
    splits = _load_and_split(cfg)
    log.info(
        "splits_loaded",
        train_rows=len(splits["train"][1]),
        val_rows=len(splits["val"][1]),
        test_rows=len(splits["test"][1]),
    )
    return train_with_splits(cfg, splits)


def train_with_splits(cfg: TrainingConfig, splits: Splits) -> TrainingResult:
    _seed_everything(cfg.seed)
    _configure_mlflow(cfg)
    _log_training_start(cfg, splits)
    with mlflow.start_run(run_name=cfg.run_name) as parent:
        _log_lineage_tags(splits)
        best = _sweep_xgb(splits, cfg)
        primary = _train_and_pick_primary(best, splits, cfg)
        version = _log_artifacts(primary, splits, cfg)
        attach_alias(cfg.model_name, version, alias=cfg.candidate_alias)
        mlflow.set_tag("model_version", str(version))
        outcome = _evaluate_and_gate(primary, splits, version, cfg)
    log.info(
        "training_done",
        run_id=parent.info.run_id,
        model_version=version,
        primary=primary.family,
        val_auprc=primary.val_metrics["auprc"],
        gate_promote=outcome.decision.promote,
        gate_reason=outcome.decision.reason,
    )
    return TrainingResult(
        run_id=parent.info.run_id,
        model_version=version,
        primary=primary,
        gate=outcome,
    )


def _log_training_start(cfg: TrainingConfig, splits: Splits) -> None:
    log.info(
        "training_start",
        tracking_uri=cfg.tracking_uri,
        experiment=cfg.experiment_name,
        n_trials=cfg.optuna_n_trials,
        timeout=cfg.optuna_timeout,
        train_rows=len(splits["train"][1]),
        val_rows=len(splits["val"][1]),
        test_rows=len(splits["test"][1]),
    )


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
    candidates = _train_candidates(best, splits, cfg)
    _log_candidate_metrics(best, candidates)
    primary = _pick_primary(*candidates)
    mlflow.set_tag("primary_family", primary.family)
    return primary


def _train_candidates(
    best: BoostingHyperparams, splits: Splits, cfg: TrainingConfig
) -> tuple[ModelResult, ModelResult]:
    scale_pos_weight = compute_scale_pos_weight(splits["train"][1])
    return (
        _fit_and_evaluate_xgb(best, scale_pos_weight, splits, cfg),
        _fit_and_evaluate_lgb(best, scale_pos_weight, splits, cfg),
    )


def _log_candidate_metrics(
    best: BoostingHyperparams, candidates: tuple[ModelResult, ModelResult]
) -> None:
    mlflow.log_params({f"best_{key}": value for key, value in asdict(best).items()})
    for result in candidates:
        _log_family_metrics(result)


def _pick_primary(xgb_result: ModelResult, lgb_result: ModelResult) -> ModelResult:
    if xgb_result.val_metrics["auprc"] >= lgb_result.val_metrics["auprc"]:
        return xgb_result
    return lgb_result


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


def _log_artifacts(primary: ModelResult, splits: Splits, cfg: TrainingConfig) -> int:
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
    return _log_and_register_model(
        primary, input_example=x_train.head(5), model_name=cfg.model_name
    )


def _log_and_register_model(
    primary: ModelResult, *, input_example: pd.DataFrame, model_name: str
) -> int:
    flavor = mlflow.xgboost if primary.family == "xgboost" else mlflow.lightgbm
    info = flavor.log_model(
        primary.model,
        name="model",
        input_example=input_example,
        registered_model_name=model_name,
    )
    return int(info.registered_model_version)


def _evaluate_and_gate(
    primary: ModelResult, splits: Splits, version: int, cfg: TrainingConfig
) -> GateOutcome:
    x_val, y_val = splits["val"]
    x_test, y_test = splits["test"]
    raw_val = primary.model.predict_proba(x_val)[:, 1]
    raw_test = primary.model.predict_proba(x_test)[:, 1]

    calibration = fit_isotonic(y_val, raw_val)
    calibrated_val = calibration.calibrator.predict(raw_val)
    calibrated_test = calibration.calibrator.predict(raw_test)
    _log_calibration(calibration, y_test, calibrated_test)

    threshold = select_threshold(
        y_val,
        calibrated_val,
        matrix=cfg.cost_matrix,
        constraints=cfg.threshold_constraints,
    )
    test_cost_total = expected_cost(y_test, calibrated_test, threshold.threshold, cfg.cost_matrix)
    test_cost_per_tx = float(test_cost_total / len(y_test))
    _log_threshold(threshold, test_cost_total, test_cost_per_tx)

    challenger = GateMetrics(
        auprc=primary.test_metrics["auprc"],
        expected_cost_per_tx=test_cost_per_tx,
    )
    champion = _load_champion_metrics(cfg)
    decision = decide(challenger, champion, tolerances=cfg.gate_tolerances)
    _log_gate(decision)
    if decision.promote:
        _promote_champion(version, challenger, cfg)

    log.info(
        "gate_decision",
        promote=decision.promote,
        reason=decision.reason,
        challenger_auprc=challenger.auprc,
        challenger_cost_per_tx=challenger.expected_cost_per_tx,
        threshold=threshold.threshold,
    )
    return GateOutcome(
        threshold=threshold,
        test_expected_cost_total=test_cost_total,
        test_expected_cost_per_tx=test_cost_per_tx,
        decision=decision,
    )


def _log_calibration(
    result: CalibrationResult, y_test: pd.Series, calibrated_test: NDArray[np.float64]
) -> None:
    mlflow.log_metric("calibrated_val_brier_before", result.brier_before)
    mlflow.log_metric("calibrated_val_brier_after", result.brier_after)
    reliability = reliability_curve_figure(
        y_test, calibrated_test, title="Test reliability (calibrated)"
    )
    mlflow.log_figure(reliability, "reliability_test.png")
    log.info(
        "calibration_fitted",
        brier_before=result.brier_before,
        brier_after=result.brier_after,
    )


def _log_threshold(
    threshold: ThresholdDecision, test_cost_total: float, test_cost_per_tx: float
) -> None:
    mlflow.log_metric("threshold_value", threshold.threshold)
    mlflow.log_metric("threshold_recall_val", threshold.recall)
    mlflow.log_metric("threshold_precision_val", threshold.precision)
    mlflow.log_metric("threshold_flagged_rate_val", threshold.flagged_rate)
    mlflow.log_metric("expected_cost_total_usd_test", test_cost_total)
    mlflow.log_metric("expected_cost_per_tx_usd_test", test_cost_per_tx)
    log.info(
        "threshold_selected",
        threshold=threshold.threshold,
        recall=threshold.recall,
        precision=threshold.precision,
        flagged_rate=threshold.flagged_rate,
        test_cost_per_tx=test_cost_per_tx,
    )


def _log_gate(decision: GateDecision) -> None:
    mlflow.set_tag("gate_reason", decision.reason)
    mlflow.log_metric("gate_promote", 1.0 if decision.promote else 0.0)


def _load_champion_metrics(cfg: TrainingConfig) -> GateMetrics | None:
    champion_version = get_alias_version(cfg.model_name, cfg.champion_alias)
    if champion_version is None:
        return None
    tags = get_version_tags(cfg.model_name, champion_version)
    if CHAMPION_TAG_AUPRC not in tags or CHAMPION_TAG_COST_PER_TX not in tags:
        log.warning(
            "champion_tags_missing",
            champion_version=champion_version,
            present=sorted(tags.keys()),
        )
        return None
    auprc_value = _safe_float(tags[CHAMPION_TAG_AUPRC])
    cost_value = _safe_float(tags[CHAMPION_TAG_COST_PER_TX])
    if auprc_value is None or cost_value is None:
        log.warning(
            "champion_tags_invalid",
            champion_version=champion_version,
            auprc=tags[CHAMPION_TAG_AUPRC],
            cost=tags[CHAMPION_TAG_COST_PER_TX],
        )
        return None
    return GateMetrics(auprc=auprc_value, expected_cost_per_tx=cost_value)


def _safe_float(raw: str) -> float | None:
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def _promote_champion(version: int, challenger: GateMetrics, cfg: TrainingConfig) -> None:
    attach_alias(cfg.model_name, version, alias=cfg.champion_alias)
    write_version_tags(
        cfg.model_name,
        version,
        {
            CHAMPION_TAG_AUPRC: f"{challenger.auprc:.6f}",
            CHAMPION_TAG_COST_PER_TX: f"{challenger.expected_cost_per_tx:.6f}",
        },
    )


def main() -> None:
    print("argus training launched", flush=True)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    cfg = TrainingConfig.from_settings()
    log.info("config_loaded", experiment=cfg.experiment_name, model=cfg.model_name)
    result = run_training(cfg)
    _write_run_marker(cfg.artifacts_dir, result)


def _write_run_marker(artifacts_dir: Path, result: TrainingResult) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": result.run_id,
        "model_version": result.model_version,
        "primary": result.primary.family,
        "metrics": result.primary.val_metrics,
        "threshold": {
            "value": result.gate.threshold.threshold,
            "recall": result.gate.threshold.recall,
            "precision": result.gate.threshold.precision,
            "flagged_rate": result.gate.threshold.flagged_rate,
        },
        "test_expected_cost": {
            "total_usd": result.gate.test_expected_cost_total,
            "per_tx_usd": result.gate.test_expected_cost_per_tx,
        },
        "gate": {
            "promote": result.gate.decision.promote,
            "reason": result.gate.decision.reason,
        },
    }
    (artifacts_dir / "last_run.json").write_text(json.dumps(payload, indent=2))


def _load_section_params(name: str, path: Path = Path("params.yaml")) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        return {}
    section = data.get(name) or {}
    return section if isinstance(section, dict) else {}


def _cost_matrix_from(params: dict[str, Any]) -> CostMatrix:
    matrix_cfg = params.get("cost_matrix") or {}
    return CostMatrix(
        fn_cost_usd=float(matrix_cfg.get("fn_cost_usd", 100.0)),
        fp_cost_usd=float(matrix_cfg.get("fp_cost_usd", 5.0)),
    )


def _threshold_constraints_from(params: dict[str, Any]) -> ThresholdConstraints:
    threshold_cfg = params.get("threshold") or {}
    return ThresholdConstraints(
        recall_floor=float(threshold_cfg.get("recall_floor", 0.5)),
        alert_volume_budget=float(threshold_cfg.get("alert_volume_budget", 0.01)),
    )


def _gate_tolerances_from(params: dict[str, Any]) -> GateTolerances:
    gate_cfg = params.get("gate") or {}
    return GateTolerances(
        auprc_tolerance=float(gate_cfg.get("auprc_tolerance", 0.0)),
        cost_tolerance=float(gate_cfg.get("cost_tolerance", 0.0)),
    )


if __name__ == "__main__":
    main()
