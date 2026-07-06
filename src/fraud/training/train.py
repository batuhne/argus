"""Train the candidates, evaluate and calibrate, gate against the champion, and register."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import mlflow
import mlflow.catboost
import mlflow.lightgbm
import mlflow.xgboost
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from fraud.common.lineage import collect_lineage, sha256_file
from fraud.common.logging import configure_logging, get_logger
from fraud.common.seed import set_seed
from fraud.config import get_settings
from fraud.dataset import add_encoded_categoricals, load_splits
from fraud.evaluation.business import CostMatrix, expected_cost
from fraud.evaluation.calibration import (
    CalibrationResult,
    brier_score,
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
from fraud.params import load_params
from fraud.paths import FEATURE_REPO_DIR, FEATURE_SERVICE, PROCESSED_DIR
from fraud.registry import (
    ARTIFACT_SHA256_TAG_CALIBRATOR,
    ARTIFACT_SHA256_TAG_ENCODER,
    BASELINE_ARTIFACT_DIR,
    CALIBRATOR_ARTIFACT_DIR,
    CHAMPION_TAG_AUPRC,
    CHAMPION_TAG_COST_PER_TX,
    CHAMPION_TAG_FAMILY,
    CHAMPION_TAG_THRESHOLD,
    ENCODER_ARTIFACT_DIR,
    attach_alias,
    get_alias_version,
    get_version_tags,
    write_version_tags,
)
from fraud.training.models import (
    BoostingHyperparams,
    build_cat,
    build_lgb,
    build_xgb,
    compute_scale_pos_weight,
)
from fraud.training.tune import tune_xgb
from fraud.transforms.encoders import CategoricalEncoder, save_encoder
from fraud.transforms.features import build_xy

Splits = dict[str, tuple[pd.DataFrame, pd.Series]]
BASELINE_SAMPLE_SIZE = 50000
# Fraction of val held out for threshold selection; the rest fits the calibrator.
_THRESHOLD_SELECTION_FRACTION = 0.5
log = get_logger(__name__)
_CALIBRATORS = {"isotonic": fit_isotonic}


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int
    tracking_uri: str
    experiment_name: str
    model_name: str
    candidate_alias: str
    champion_alias: str
    optuna_n_trials: int
    optuna_timeout: int
    encoder_smoothing: float
    encoder_n_splits: int
    shap_sample_size: int
    recall_at_k_levels: tuple[float, ...]
    calibration_method: str
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
        params = load_params()
        training = params.training
        evaluation = params.evaluation
        return cls(
            seed=params.seed,
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow_experiment_name,
            model_name=settings.argus_model_name,
            candidate_alias=training.candidate_alias,
            champion_alias=evaluation.champion_alias,
            optuna_n_trials=n_trials if n_trials is not None else training.optuna.n_trials,
            optuna_timeout=timeout if timeout is not None else training.optuna.timeout_seconds,
            encoder_smoothing=training.encoder.smoothing,
            encoder_n_splits=training.encoder.n_splits,
            shap_sample_size=training.shap.sample_size,
            recall_at_k_levels=evaluation.recall_at_k_levels,
            calibration_method=evaluation.calibration.method,
            cost_matrix=CostMatrix(
                fn_cost_usd=evaluation.cost_matrix.fn_cost_usd,
                fp_cost_usd=evaluation.cost_matrix.fp_cost_usd,
            ),
            threshold_constraints=ThresholdConstraints(
                recall_floor=evaluation.threshold.recall_floor,
                alert_volume_budget=evaluation.threshold.alert_volume_budget,
            ),
            gate_tolerances=GateTolerances(
                auprc_tolerance=evaluation.gate.auprc_tolerance,
                cost_tolerance=evaluation.gate.cost_tolerance,
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
class GateOutcome:
    threshold: ThresholdDecision
    test_expected_cost_total: float
    test_expected_cost_per_tx: float
    decision: GateDecision


@dataclass(frozen=True, slots=True)
class TrainingResult:
    run_id: str
    model_version: int
    champion_version: int | None
    primary: ModelResult
    gate: GateOutcome


def run_training(cfg: TrainingConfig) -> TrainingResult:
    log.info("loading_splits", repo_dir=str(cfg.repo_dir), processed_dir=str(cfg.processed_dir))
    splits, encoder = _load_and_split(cfg)
    log.info(
        "splits_loaded",
        train_rows=len(splits["train"][1]),
        val_rows=len(splits["val"][1]),
        test_rows=len(splits["test"][1]),
    )
    return train_with_splits(cfg, splits, encoder)


def train_with_splits(
    cfg: TrainingConfig, splits: Splits, encoder: CategoricalEncoder
) -> TrainingResult:
    _seed_everything(cfg.seed)
    _configure_mlflow(cfg)
    _log_training_start(cfg, splits)
    with mlflow.start_run(run_name=cfg.run_name) as parent:
        _log_lineage_tags(splits)
        _log_run_constraints(cfg)
        best = _sweep_xgb(splits, cfg)
        primary = _train_and_pick_primary(best, splits, cfg)
        version = _log_artifacts(primary, splits, cfg, encoder)
        attach_alias(cfg.model_name, version, alias=cfg.candidate_alias)
        mlflow.set_tag("model_version", str(version))
        prior_champion = get_alias_version(cfg.model_name, cfg.champion_alias)
        outcome = _evaluate_and_gate(primary, splits, version, cfg)
    # compute, don't re-read: a concurrent alias move would misattribute the champion
    champion_version = version if outcome.decision.promote else prior_champion
    log.info(
        "training_done",
        run_id=parent.info.run_id,
        model_version=version,
        champion_version=champion_version,
        primary=primary.family,
        val_auprc=primary.val_metrics["auprc"],
        gate_promote=outcome.decision.promote,
        gate_reason=outcome.decision.reason,
    )
    return TrainingResult(
        run_id=parent.info.run_id,
        model_version=version,
        champion_version=champion_version,
        primary=primary,
        gate=outcome,
    )


def _log_run_constraints(cfg: TrainingConfig) -> None:
    # Threshold and gate constraints are what make a run's decision reproducible.
    mlflow.log_params(
        {
            "recall_floor": cfg.threshold_constraints.recall_floor,
            "alert_volume_budget": cfg.threshold_constraints.alert_volume_budget,
            "auprc_tolerance": cfg.gate_tolerances.auprc_tolerance,
            "cost_tolerance": cfg.gate_tolerances.cost_tolerance,
            "fn_cost_usd": cfg.cost_matrix.fn_cost_usd,
            "fp_cost_usd": cfg.cost_matrix.fp_cost_usd,
        }
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


def _load_and_split(cfg: TrainingConfig) -> tuple[Splits, CategoricalEncoder]:
    frames = load_splits(cfg.repo_dir, cfg.processed_dir)
    encoder = add_encoded_categoricals(
        frames,
        seed=cfg.seed,
        smoothing=cfg.encoder_smoothing,
        n_splits=cfg.encoder_n_splits,
    )
    return {split: build_xy(frame) for split, frame in frames.items()}, encoder


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
) -> tuple[ModelResult, ModelResult, ModelResult]:
    scale_pos_weight = compute_scale_pos_weight(splits["train"][1])
    return (
        _fit_and_evaluate_xgb(best, scale_pos_weight, splits, cfg),
        _fit_and_evaluate_lgb(best, scale_pos_weight, splits, cfg),
        _fit_and_evaluate_cat(best, scale_pos_weight, splits, cfg),
    )


def _log_candidate_metrics(best: BoostingHyperparams, candidates: tuple[ModelResult, ...]) -> None:
    mlflow.log_params({f"best_{key}": value for key, value in asdict(best).items()})
    for result in candidates:
        _log_family_metrics(result)


def _pick_primary(*candidates: ModelResult) -> ModelResult:
    # Highest validation AUPRC wins; ties keep the earliest candidate, so xgboost is preferred.
    # A degenerate NaN score sinks to -inf so it can never outrank a real candidate.
    def val_auprc(result: ModelResult) -> float:
        score = result.val_metrics["auprc"]
        return score if np.isfinite(score) else float("-inf")

    return max(candidates, key=val_auprc)


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


def _fit_and_evaluate_cat(
    best: BoostingHyperparams, scale_pos_weight: float, splits: Splits, cfg: TrainingConfig
) -> ModelResult:
    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]
    model = build_cat(best, scale_pos_weight=scale_pos_weight, seed=cfg.seed)
    model.fit(x_train, y_train, eval_set=(x_val, y_val), verbose=False)
    metrics = _evaluate_on_splits(model, splits, cfg.recall_at_k_levels)
    return ModelResult("catboost", model, *metrics)


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


def _log_artifacts(
    primary: ModelResult, splits: Splits, cfg: TrainingConfig, encoder: CategoricalEncoder
) -> int:
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
    _log_baseline_artifact(x_train, cfg.seed)
    _log_encoder_artifact(encoder)
    return _log_and_register_model(
        primary, input_example=x_train.head(5), model_name=cfg.model_name
    )


def _log_encoder_artifact(encoder: CategoricalEncoder) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "encoder.joblib"
        save_encoder(encoder, path)
        mlflow.set_tag(ARTIFACT_SHA256_TAG_ENCODER, sha256_file(path))
        mlflow.log_artifact(str(path), artifact_path=ENCODER_ARTIFACT_DIR)


def _log_and_register_model(
    primary: ModelResult, *, input_example: pd.DataFrame, model_name: str
) -> int:
    info = _model_flavor(primary.family).log_model(
        primary.model,
        name="model",
        input_example=input_example,
        registered_model_name=model_name,
    )
    return int(info.registered_model_version)


def _model_flavor(family: str) -> Any:
    flavors = {
        "xgboost": mlflow.xgboost,
        "lightgbm": mlflow.lightgbm,
        "catboost": mlflow.catboost,
    }
    if family not in flavors:
        raise RuntimeError(f"unsupported model family {family!r}")
    return flavors[family]


def _evaluate_and_gate(
    primary: ModelResult, splits: Splits, version: int, cfg: TrainingConfig
) -> GateOutcome:
    x_val, y_val = splits["val"]
    x_test, y_test = splits["test"]
    raw_val = primary.model.predict_proba(x_val)[:, 1]
    raw_test = primary.model.predict_proba(x_test)[:, 1]

    fit_idx, select_idx = _split_val_for_calibration(y_val, cfg.seed)
    calibration = _CALIBRATORS[cfg.calibration_method](y_val.iloc[fit_idx], raw_val[fit_idx])
    calibrated_select = calibration.calibrator.predict(raw_val[select_idx])
    calibrated_test = calibration.calibrator.predict(raw_test)
    _log_calibration(calibration, y_test, raw_test, calibrated_test)

    threshold = select_threshold(
        y_val.iloc[select_idx],
        calibrated_select,
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
        _promote_champion(
            version, challenger, cfg, threshold=threshold.threshold, family=primary.family
        )

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


def _split_val_for_calibration(
    y_val: pd.Series, seed: int
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Split val so the threshold is not chosen on the calibrator's own fit data."""
    positions = np.arange(len(y_val))
    fit_idx, select_idx = train_test_split(
        positions,
        test_size=_THRESHOLD_SELECTION_FRACTION,
        random_state=seed,
        stratify=y_val.to_numpy(),
    )
    return fit_idx, select_idx


def _log_calibration(
    result: CalibrationResult,
    y_test: pd.Series,
    raw_test: NDArray[np.float64],
    calibrated_test: NDArray[np.float64],
) -> None:
    brier_before = brier_score(y_test, raw_test)
    brier_after = brier_score(y_test, calibrated_test)
    mlflow.log_metric("calibrated_test_brier_before", brier_before)
    mlflow.log_metric("calibrated_test_brier_after", brier_after)
    reliability = reliability_curve_figure(
        y_test, calibrated_test, title="Test reliability (calibrated)"
    )
    mlflow.log_figure(reliability, "reliability_test.png")
    _log_calibrator_artifact(result)
    log.info("calibration_fitted", brier_before=brier_before, brier_after=brier_after)


def _log_calibrator_artifact(result: CalibrationResult) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "calibrator.joblib"
        joblib.dump(result.calibrator, path)
        mlflow.set_tag(ARTIFACT_SHA256_TAG_CALIBRATOR, sha256_file(path))
        mlflow.log_artifact(str(path), artifact_path=CALIBRATOR_ARTIFACT_DIR)


def _log_baseline_artifact(x_train: pd.DataFrame, seed: int) -> None:
    """Freeze a training feature sample as the drift-monitoring reference."""
    sample_size = min(BASELINE_SAMPLE_SIZE, len(x_train))
    baseline = x_train.sample(sample_size, random_state=seed)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "baseline.parquet"
        baseline.to_parquet(path, index=False)
        mlflow.log_artifact(str(path), artifact_path=BASELINE_ARTIFACT_DIR)


def _log_threshold(
    threshold: ThresholdDecision, test_cost_total: float, test_cost_per_tx: float
) -> None:
    mlflow.log_metric("threshold_value", threshold.threshold)
    mlflow.log_metric("threshold_recall_val_select", threshold.recall)
    mlflow.log_metric("threshold_precision_val_select", threshold.precision)
    mlflow.log_metric("threshold_flagged_rate_val_select", threshold.flagged_rate)
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
        raise RuntimeError(
            f"champion version {champion_version} is missing gate tags {sorted(tags.keys())}; "
            "refusing to treat a corrupt champion as an open bootstrap"
        )
    auprc_value = _safe_float(tags[CHAMPION_TAG_AUPRC])
    cost_value = _safe_float(tags[CHAMPION_TAG_COST_PER_TX])
    if auprc_value is None or cost_value is None:
        raise RuntimeError(
            f"champion version {champion_version} has non-finite gate tags "
            f"(auprc={tags[CHAMPION_TAG_AUPRC]!r}, cost={tags[CHAMPION_TAG_COST_PER_TX]!r})"
        )
    return GateMetrics(auprc=auprc_value, expected_cost_per_tx=cost_value)


def _safe_float(raw: str) -> float | None:
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def _promote_champion(
    version: int,
    challenger: GateMetrics,
    cfg: TrainingConfig,
    *,
    threshold: float,
    family: str,
) -> None:
    # Tags first, alias last: a partial tag write then leaves the old champion whole and servable.
    write_version_tags(
        cfg.model_name,
        version,
        {
            CHAMPION_TAG_AUPRC: f"{challenger.auprc:.6f}",
            CHAMPION_TAG_COST_PER_TX: f"{challenger.expected_cost_per_tx:.6f}",
            CHAMPION_TAG_THRESHOLD: f"{threshold:.6f}",
            CHAMPION_TAG_FAMILY: family,
        },
    )
    attach_alias(cfg.model_name, version, alias=cfg.champion_alias)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    log.info("training_launched")
    cfg = TrainingConfig.from_settings()
    log.info("config_loaded", experiment=cfg.experiment_name, model=cfg.model_name)
    result = run_training(cfg)
    _write_run_marker(cfg.artifacts_dir, result)


def _write_run_marker(artifacts_dir: Path, result: TrainingResult) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": result.run_id,
        "model_version": result.model_version,
        "champion_version": result.champion_version,
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
    tmp = artifacts_dir / "last_run.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(artifacts_dir / "last_run.json")


if __name__ == "__main__":
    main()
