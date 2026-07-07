"""Load the champion bundle (model, calibrator, encoder, threshold) from the MLflow registry."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.catboost
import mlflow.lightgbm
import mlflow.xgboost
from mlflow.exceptions import MlflowException

from fraud.calibrator import IsotonicCalibrator
from fraud.common.lineage import sha256_file
from fraud.common.logging import get_logger
from fraud.config import get_settings
from fraud.params import load_params
from fraud.registry import (
    ARTIFACT_SHA256_TAG_CALIBRATOR,
    ARTIFACT_SHA256_TAG_ENCODER,
    CALIBRATOR_ARTIFACT_PATH,
    CHAMPION_TAG_FAMILY,
    CHAMPION_TAG_THRESHOLD,
    ENCODER_ARTIFACT_PATH,
    get_alias_version,
    get_run_tags,
    get_version_run_id,
    get_version_tags,
)
from fraud.transforms.encoders import CategoricalEncoder, load_encoder
from fraud.transforms.features import FEATURE_COLUMNS

log = get_logger(__name__)

_RETRY_BASE_SECONDS = 1.0
_RETRY_CAP_SECONDS = 8.0


@dataclass(frozen=True, slots=True)
class ChampionLoadConfig:
    tracking_uri: str
    model_name: str
    champion_alias: str
    request_timeout_seconds: int
    load_deadline_seconds: float

    @classmethod
    def from_settings(cls) -> ChampionLoadConfig:
        settings = get_settings()
        return cls(
            tracking_uri=settings.mlflow_tracking_uri,
            model_name=settings.argus_model_name,
            champion_alias=load_params().evaluation.champion_alias,
            request_timeout_seconds=settings.mlflow_request_timeout_seconds,
            load_deadline_seconds=settings.model_load_deadline_seconds,
        )


@dataclass(frozen=True, slots=True)
class ModelBundle:
    model: Any
    calibrator: IsotonicCalibrator
    encoder: CategoricalEncoder
    threshold: float
    version: int
    family: str


class ChampionUnavailableError(RuntimeError):
    """Champion alias or tags not in the registry yet; a cold start may be racing training."""


class ArtifactIntegrityError(RuntimeError):
    """A pickled artifact's hash does not match the value recorded at training time."""


class FeatureContractError(RuntimeError):
    """The champion model's feature names do not match the serving feature contract."""


def load_champion(cfg: ChampionLoadConfig) -> ModelBundle:
    """Load the champion bundle, retrying a not-yet-ready registry until the deadline."""
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", str(cfg.request_timeout_seconds))
    mlflow.set_tracking_uri(cfg.tracking_uri)
    deadline = time.monotonic() + cfg.load_deadline_seconds
    attempt = 0
    while True:
        try:
            return _load_bundle(cfg)
        except (ChampionUnavailableError, MlflowException, OSError) as exc:
            if time.monotonic() >= deadline:
                log.error("champion_load_failed", attempts=attempt + 1, error=str(exc))
                raise
            delay = _backoff_delay(attempt)
            log.warning("champion_load_retry", attempt=attempt, error=str(exc), retry_in=delay)
            time.sleep(delay)
            attempt += 1


def _backoff_delay(attempt: int) -> float:
    capped = min(_RETRY_CAP_SECONDS, _RETRY_BASE_SECONDS * 2.0**attempt)
    return capped + random.uniform(0.0, _RETRY_BASE_SECONDS)


def _load_bundle(cfg: ChampionLoadConfig) -> ModelBundle:
    version = get_alias_version(cfg.model_name, cfg.champion_alias)
    if version is None:
        raise ChampionUnavailableError(
            f"no '{cfg.champion_alias}' alias on model '{cfg.model_name}'; train and promote first"
        )

    tags = get_version_tags(cfg.model_name, version)
    threshold = _required_threshold(tags, version)
    family = tags.get(CHAMPION_TAG_FAMILY, "")
    if not family:
        raise ChampionUnavailableError(
            f"champion version {version} has no '{CHAMPION_TAG_FAMILY}' tag yet"
        )

    # Pin the resolved version so the model matches its version-loaded calibrator and encoder.
    model_uri = f"models:/{cfg.model_name}/{version}"
    model = _load_model_by_family(family, model_uri)
    _verify_feature_contract(family, model)
    run_id = get_version_run_id(cfg.model_name, version)
    run_tags = get_run_tags(run_id)
    calibrator = _load_calibrator(run_id, run_tags)
    encoder = _load_encoder(run_id, run_tags)
    return ModelBundle(model, calibrator, encoder, threshold, version, family)


def _required_threshold(tags: dict[str, str], version: int) -> float:
    raw = tags.get(CHAMPION_TAG_THRESHOLD)
    if raw is None:
        raise ChampionUnavailableError(
            f"champion version {version} has no '{CHAMPION_TAG_THRESHOLD}' tag yet"
        )
    try:
        value = float(raw)
    except ValueError as exc:
        raise ChampionUnavailableError(
            f"champion version {version} has a non-numeric '{CHAMPION_TAG_THRESHOLD}' tag {raw!r}"
        ) from exc
    if not 0.0 <= value <= 1.0:
        raise ChampionUnavailableError(
            f"champion version {version} has an out-of-range decision_threshold {value}"
        )
    return value


def _load_model_by_family(family: str, model_uri: str) -> Any:
    if family == "xgboost":
        return mlflow.xgboost.load_model(model_uri)
    if family == "lightgbm":
        return mlflow.lightgbm.load_model(model_uri)
    if family == "catboost":
        return mlflow.catboost.load_model(model_uri)
    raise RuntimeError(f"unsupported champion family {family!r}")


def _verify_feature_contract(family: str, model: Any) -> None:
    names = _model_feature_names(family, model)
    if not names:
        raise FeatureContractError(
            f"champion ({family}) exposes no feature names to check against the serving contract"
        )
    if set(names) != set(FEATURE_COLUMNS):
        missing = sorted(set(FEATURE_COLUMNS) - set(names))
        extra = sorted(set(names) - set(FEATURE_COLUMNS))
        raise FeatureContractError(
            f"champion feature set does not match the serving contract; "
            f"missing={missing} unexpected={extra}"
        )


def _model_feature_names(family: str, model: Any) -> tuple[str, ...]:
    if family == "xgboost":
        names = model.get_booster().feature_names
    elif family == "lightgbm":
        names = model.feature_name_
    elif family == "catboost":
        names = model.feature_names_
    else:
        names = None
    return tuple(names) if names else ()


def _load_calibrator(run_id: str, run_tags: dict[str, str]) -> IsotonicCalibrator:
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=CALIBRATOR_ARTIFACT_PATH
    )
    _verify_artifact_integrity(
        Path(local_path), run_tags, ARTIFACT_SHA256_TAG_CALIBRATOR, "calibrator"
    )
    calibrator: IsotonicCalibrator = joblib.load(local_path)
    return calibrator


def _load_encoder(run_id: str, run_tags: dict[str, str]) -> CategoricalEncoder:
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=ENCODER_ARTIFACT_PATH
    )
    _verify_artifact_integrity(Path(local_path), run_tags, ARTIFACT_SHA256_TAG_ENCODER, "encoder")
    return load_encoder(Path(local_path))


def _verify_artifact_integrity(
    path: Path, run_tags: dict[str, str], tag_key: str, name: str
) -> None:
    # Raises outside the load-retry path so a tampered or unhashed artifact fails fast.
    expected = run_tags.get(tag_key)
    if not expected:
        raise ArtifactIntegrityError(
            f"champion run has no '{tag_key}' tag; retrain to record the {name} artifact hash "
            "before it can be loaded"
        )
    actual = sha256_file(path)
    if actual != expected:
        raise ArtifactIntegrityError(
            f"{name} artifact hash {actual} does not match the recorded {expected}; "
            "refusing to unpickle a tampered artifact"
        )
