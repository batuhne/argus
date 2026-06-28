import dataclasses
import time
from typing import Any

import mlflow.catboost
import mlflow.lightgbm
import mlflow.xgboost
import pytest

from fraud.serving.config import ServingConfig
from fraud.serving.model import (
    _RETRY_BASE_SECONDS,
    _RETRY_CAP_SECONDS,
    ChampionUnavailableError,
    _backoff_delay,
    _load_model_by_family,
    load_champion,
)


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


def test_serving_config_sources_champion_alias_from_params() -> None:
    from fraud.params import load_params

    assert ServingConfig.from_settings().champion_alias == load_params().evaluation.champion_alias


def _cfg(**overrides: Any) -> ServingConfig:
    return dataclasses.replace(ServingConfig.from_settings(), **overrides)


def test_load_champion_retries_until_registry_is_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = object()
    calls = {"n": 0}

    def fake_load(_cfg: ServingConfig) -> object:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ChampionUnavailableError("not yet")
        return bundle

    monkeypatch.setattr("fraud.serving.model._load_bundle", fake_load)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    assert load_champion(_cfg(load_deadline_seconds=30.0)) is bundle
    assert calls["n"] == 3


def test_load_champion_gives_up_after_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_unavailable(_cfg: ServingConfig) -> object:
        raise ChampionUnavailableError("never ready")

    monkeypatch.setattr("fraud.serving.model._load_bundle", always_unavailable)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with pytest.raises(ChampionUnavailableError):
        load_champion(_cfg(load_deadline_seconds=0.0))


def test_load_champion_does_not_retry_misconfiguration(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def bad_family(_cfg: ServingConfig) -> object:
        calls["n"] += 1
        raise RuntimeError("unsupported champion family 'rf'")

    monkeypatch.setattr("fraud.serving.model._load_bundle", bad_family)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match="unsupported champion family"):
        load_champion(_cfg(load_deadline_seconds=30.0))
    assert calls["n"] == 1


def test_backoff_delay_increases_and_is_capped() -> None:
    assert _RETRY_BASE_SECONDS <= _backoff_delay(0) <= 2 * _RETRY_BASE_SECONDS
    assert _RETRY_CAP_SECONDS <= _backoff_delay(20) <= _RETRY_CAP_SECONDS + _RETRY_BASE_SECONDS
