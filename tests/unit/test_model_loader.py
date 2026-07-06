import dataclasses
import time
from pathlib import Path
from typing import Any

import mlflow.catboost
import mlflow.lightgbm
import mlflow.xgboost
import pytest

from fraud.common.lineage import sha256_file
from fraud.model_loader import (
    _RETRY_BASE_SECONDS,
    _RETRY_CAP_SECONDS,
    ArtifactIntegrityError,
    ChampionLoadConfig,
    ChampionUnavailableError,
    FeatureContractError,
    _backoff_delay,
    _load_model_by_family,
    _required_threshold,
    _verify_artifact_integrity,
    _verify_feature_contract,
    load_champion,
)
from fraud.registry import ARTIFACT_SHA256_TAG_CALIBRATOR, CHAMPION_TAG_THRESHOLD
from fraud.serving.config import ServingConfig
from fraud.transforms.features import FEATURE_COLUMNS


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


def test_champion_load_config_sources_alias_from_params() -> None:
    from fraud.params import load_params

    assert (
        ChampionLoadConfig.from_settings().champion_alias == load_params().evaluation.champion_alias
    )


def _cfg(**overrides: Any) -> ChampionLoadConfig:
    return dataclasses.replace(ChampionLoadConfig.from_settings(), **overrides)


def test_load_champion_retries_until_registry_is_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = object()
    calls = {"n": 0}

    def fake_load(_cfg: ChampionLoadConfig) -> object:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ChampionUnavailableError("not yet")
        return bundle

    monkeypatch.setattr("fraud.model_loader._load_bundle", fake_load)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    assert load_champion(_cfg(load_deadline_seconds=30.0)) is bundle
    assert calls["n"] == 3


def test_load_champion_gives_up_after_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_unavailable(_cfg: ChampionLoadConfig) -> object:
        raise ChampionUnavailableError("never ready")

    monkeypatch.setattr("fraud.model_loader._load_bundle", always_unavailable)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with pytest.raises(ChampionUnavailableError):
        load_champion(_cfg(load_deadline_seconds=0.0))


def test_load_champion_does_not_retry_misconfiguration(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def bad_family(_cfg: ChampionLoadConfig) -> object:
        calls["n"] += 1
        raise RuntimeError("unsupported champion family 'rf'")

    monkeypatch.setattr("fraud.model_loader._load_bundle", bad_family)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match="unsupported champion family"):
        load_champion(_cfg(load_deadline_seconds=30.0))
    assert calls["n"] == 1


def test_load_champion_does_not_retry_an_integrity_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    def tampered(_cfg: ChampionLoadConfig) -> object:
        calls["n"] += 1
        raise ArtifactIntegrityError("calibrator artifact hash does not match the recorded")

    monkeypatch.setattr("fraud.model_loader._load_bundle", tampered)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with pytest.raises(ArtifactIntegrityError):
        load_champion(_cfg(load_deadline_seconds=30.0))
    assert calls["n"] == 1


def test_backoff_delay_increases_and_is_capped() -> None:
    assert _RETRY_BASE_SECONDS <= _backoff_delay(0) <= 2 * _RETRY_BASE_SECONDS
    assert _RETRY_CAP_SECONDS <= _backoff_delay(20) <= _RETRY_CAP_SECONDS + _RETRY_BASE_SECONDS


def test_required_threshold_returns_a_valid_value() -> None:
    assert _required_threshold({CHAMPION_TAG_THRESHOLD: "0.5"}, 1) == 0.5


def test_required_threshold_rejects_a_missing_tag() -> None:
    with pytest.raises(ChampionUnavailableError, match="decision_threshold"):
        _required_threshold({}, 1)


@pytest.mark.parametrize("raw", ["", "tbd", "0.5x"])
def test_required_threshold_rejects_a_non_numeric_tag(raw: str) -> None:
    with pytest.raises(ChampionUnavailableError, match="non-numeric"):
        _required_threshold({CHAMPION_TAG_THRESHOLD: raw}, 1)


@pytest.mark.parametrize("raw", ["-0.1", "1.5", "nan", "inf"])
def test_required_threshold_rejects_out_of_range_or_non_finite(raw: str) -> None:
    with pytest.raises(ChampionUnavailableError, match="out-of-range"):
        _required_threshold({CHAMPION_TAG_THRESHOLD: raw}, 1)


def test_verify_artifact_integrity_accepts_a_matching_hash(tmp_path: Path) -> None:
    path = tmp_path / "artifact.joblib"
    path.write_bytes(b"model-bytes")
    tags = {ARTIFACT_SHA256_TAG_CALIBRATOR: sha256_file(path)}
    _verify_artifact_integrity(path, tags, ARTIFACT_SHA256_TAG_CALIBRATOR, "calibrator")


def test_verify_artifact_integrity_rejects_a_tampered_artifact(tmp_path: Path) -> None:
    path = tmp_path / "artifact.joblib"
    path.write_bytes(b"model-bytes")
    tags = {ARTIFACT_SHA256_TAG_CALIBRATOR: sha256_file(path)}
    path.write_bytes(b"swapped-bytes")
    with pytest.raises(ArtifactIntegrityError, match="tampered"):
        _verify_artifact_integrity(path, tags, ARTIFACT_SHA256_TAG_CALIBRATOR, "calibrator")


def test_verify_artifact_integrity_rejects_a_missing_hash_tag(tmp_path: Path) -> None:
    path = tmp_path / "artifact.joblib"
    path.write_bytes(b"model-bytes")
    with pytest.raises(ArtifactIntegrityError, match="retrain"):
        _verify_artifact_integrity(path, {}, ARTIFACT_SHA256_TAG_CALIBRATOR, "calibrator")


class _NamedModel:
    def __init__(self, names: list[str]) -> None:
        self.feature_name_ = names


def test_feature_contract_accepts_the_serving_columns() -> None:
    _verify_feature_contract("lightgbm", _NamedModel(list(FEATURE_COLUMNS)))


def test_feature_contract_rejects_a_mismatched_model() -> None:
    with pytest.raises(FeatureContractError, match="unexpected"):
        _verify_feature_contract("lightgbm", _NamedModel([*FEATURE_COLUMNS, "stowaway"]))


def test_feature_contract_skips_when_names_are_absent() -> None:
    _verify_feature_contract("lightgbm", _NamedModel([]))
