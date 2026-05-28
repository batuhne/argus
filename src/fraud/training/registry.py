from __future__ import annotations

from mlflow import MlflowClient


def attach_alias(model_name: str, version: int, *, alias: str = "candidate") -> None:
    """Point a registry alias at the given model version."""
    MlflowClient().set_registered_model_alias(model_name, alias, str(version))
