from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.catboost
import mlflow.lightgbm
import mlflow.xgboost

from fraud.evaluation.calibration import IsotonicCalibrator
from fraud.serving.config import ServingConfig
from fraud.training.registry import (
    CALIBRATOR_ARTIFACT_PATH,
    CHAMPION_TAG_FAMILY,
    CHAMPION_TAG_THRESHOLD,
    ENCODER_ARTIFACT_PATH,
    get_alias_version,
    get_version_run_id,
    get_version_tags,
)
from fraud.transforms.encoders import CategoricalEncoder, load_encoder


@dataclass(frozen=True, slots=True)
class ModelBundle:
    model: Any
    calibrator: IsotonicCalibrator
    encoder: CategoricalEncoder
    threshold: float
    version: int
    family: str


def load_champion(cfg: ServingConfig) -> ModelBundle:
    """Resolve the champion alias and load its model, calibrator, and threshold."""
    mlflow.set_tracking_uri(cfg.tracking_uri)
    version = get_alias_version(cfg.model_name, cfg.champion_alias)
    if version is None:
        raise RuntimeError(
            f"no '{cfg.champion_alias}' alias on model '{cfg.model_name}'; train and promote first"
        )

    tags = get_version_tags(cfg.model_name, version)
    threshold = _required_threshold(tags, version)
    family = tags.get(CHAMPION_TAG_FAMILY, "")

    model_uri = f"models:/{cfg.model_name}@{cfg.champion_alias}"
    model = _load_model_by_family(family, model_uri)
    calibrator = _load_calibrator(cfg.model_name, version)
    encoder = _load_encoder(cfg.model_name, version)
    return ModelBundle(model, calibrator, encoder, threshold, version, family)


def _required_threshold(tags: dict[str, str], version: int) -> float:
    raw = tags.get(CHAMPION_TAG_THRESHOLD)
    if raw is None:
        raise RuntimeError(
            f"champion version {version} has no '{CHAMPION_TAG_THRESHOLD}' tag; "
            "retrain to persist it"
        )
    return float(raw)


def _load_model_by_family(family: str, model_uri: str) -> Any:
    if family == "xgboost":
        return mlflow.xgboost.load_model(model_uri)
    if family == "lightgbm":
        return mlflow.lightgbm.load_model(model_uri)
    if family == "catboost":
        return mlflow.catboost.load_model(model_uri)
    raise RuntimeError(f"unsupported champion family {family!r}")


def _load_calibrator(model_name: str, version: int) -> IsotonicCalibrator:
    run_id = get_version_run_id(model_name, version)
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=CALIBRATOR_ARTIFACT_PATH
    )
    calibrator: IsotonicCalibrator = joblib.load(local_path)
    return calibrator


def _load_encoder(model_name: str, version: int) -> CategoricalEncoder:
    run_id = get_version_run_id(model_name, version)
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=ENCODER_ARTIFACT_PATH
    )
    return load_encoder(Path(local_path))
