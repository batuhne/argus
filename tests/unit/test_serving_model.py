import mlflow.catboost
import mlflow.lightgbm
import mlflow.xgboost
import pytest

from fraud.serving.model import _load_model_by_family


def test_load_model_by_family_dispatches_to_each_flavor(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinels = {family: object() for family in ("xgboost", "lightgbm", "catboost")}
    monkeypatch.setattr(mlflow.xgboost, "load_model", lambda uri: sentinels["xgboost"])
    monkeypatch.setattr(mlflow.lightgbm, "load_model", lambda uri: sentinels["lightgbm"])
    monkeypatch.setattr(mlflow.catboost, "load_model", lambda uri: sentinels["catboost"])

    for family, sentinel in sentinels.items():
        assert _load_model_by_family(family, "models:/m@champion") is sentinel


def test_load_model_by_family_rejects_unknown_family() -> None:
    with pytest.raises(RuntimeError, match="unsupported champion family"):
        _load_model_by_family("randomforest", "models:/m@champion")
