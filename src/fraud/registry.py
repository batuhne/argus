"""MLflow registry helpers, plus the artifact-path and champion-tag key constants."""

from __future__ import annotations

from collections.abc import Mapping

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

CHAMPION_TAG_AUPRC = "test_auprc"
CHAMPION_TAG_COST_PER_TX = "expected_cost_per_tx_usd"
CHAMPION_TAG_THRESHOLD = "decision_threshold"
CHAMPION_TAG_FAMILY = "primary_family"
CALIBRATOR_ARTIFACT_DIR = "calibrator"
CALIBRATOR_ARTIFACT_PATH = f"{CALIBRATOR_ARTIFACT_DIR}/calibrator.joblib"
ENCODER_ARTIFACT_DIR = "encoder"
ENCODER_ARTIFACT_PATH = f"{ENCODER_ARTIFACT_DIR}/encoder.joblib"
BASELINE_ARTIFACT_DIR = "monitoring"
BASELINE_ARTIFACT_PATH = f"{BASELINE_ARTIFACT_DIR}/baseline.parquet"


def attach_alias(model_name: str, version: int, *, alias: str = "candidate") -> None:
    """Point a registry alias at the given model version."""
    MlflowClient().set_registered_model_alias(model_name, alias, str(version))


def get_alias_version(model_name: str, alias: str) -> int | None:
    """Return the version pointed at by the alias, or None if the model or alias is unset."""
    try:
        registered = MlflowClient().get_registered_model(model_name)
    except MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise
    aliases: Mapping[str, str] = registered.aliases or {}
    raw = aliases.get(alias)
    return int(raw) if raw is not None else None


def get_version_tags(model_name: str, version: int) -> dict[str, str]:
    """Return the user tags stored on a registered model version."""
    mv = MlflowClient().get_model_version(model_name, str(version))
    return dict(mv.tags or {})


def get_version_run_id(model_name: str, version: int) -> str:
    """Return the source run id of a registered model version."""
    run_id = MlflowClient().get_model_version(model_name, str(version)).run_id
    if not run_id:
        raise RuntimeError(f"model '{model_name}' version {version} has no source run id")
    return str(run_id)


def write_version_tags(model_name: str, version: int, tags: Mapping[str, str]) -> None:
    """Persist evaluation metrics on a registered model version for fast champion lookup."""
    client = MlflowClient()
    for key, value in tags.items():
        client.set_model_version_tag(model_name, str(version), key, value)
