from __future__ import annotations

from mlflow import MlflowClient, register_model


def register_candidate(
    run_id: str,
    model_name: str,
    *,
    artifact_name: str = "model",
    alias: str = "candidate",
) -> int:
    """Register the model from a run and point an alias at the new version."""
    model_uri = f"runs:/{run_id}/{artifact_name}"
    version = register_model(model_uri, model_name).version
    MlflowClient().set_registered_model_alias(model_name, alias, version)
    return int(version)
