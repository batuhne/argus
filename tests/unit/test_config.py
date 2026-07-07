from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from fraud.config import Settings, get_settings


def test_settings_has_expected_shape() -> None:
    settings = Settings()
    assert settings.seed == 42
    assert isinstance(settings.redis_port, int)
    assert settings.mlflow_tracking_uri.startswith("http")


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_serving_api_key_defaults_to_none() -> None:
    assert Settings.model_fields["serving_api_key"].default is None


def test_environment_defaults_to_production() -> None:
    # Fail closed by default: an unconfigured deploy must not silently open the auth boundary.
    assert Settings.model_fields["environment"].default == "production"


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_serving_api_key_coerces_to_none(blank: str) -> None:
    assert Settings(serving_api_key=SecretStr(blank)).serving_api_key is None


def test_blank_serving_api_key_from_env_coerces_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERVING_API_KEY", "")
    assert Settings().serving_api_key is None


def test_serving_api_key_is_kept_when_set() -> None:
    key = Settings(serving_api_key=SecretStr("secret")).serving_api_key
    assert key is not None
    assert key.get_secret_value() == "secret"


def test_bootstrap_default_lists_every_broker() -> None:
    # The code default, not a .env override.
    default = Settings.model_fields["kafka_bootstrap_servers"].default
    brokers = default.split(",")
    assert len(brokers) == 3
    assert all(":" in broker for broker in brokers)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", -1),
        ("mlflow_request_timeout_seconds", 0),
        ("model_load_deadline_seconds", 0.0),
        ("champion_reload_interval_seconds", -1.0),
        ("monitoring_exporter_port", 0),
        ("consumer_metrics_port", 70000),
        ("retrain_trigger_metrics_port", -1),
        ("redis_port", 0),
    ],
)
def test_out_of_range_settings_are_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})


def test_reload_interval_zero_is_allowed_to_disable() -> None:
    # 0 disables the hot-reload loop, so the bound permits it while still rejecting negatives.
    assert Settings(champion_reload_interval_seconds=0.0).champion_reload_interval_seconds == 0.0


@pytest.mark.parametrize("url", ["http://localhost:3001/predict", "https://serving:3001/predict"])
def test_serving_predict_url_accepts_http_schemes(url: str) -> None:
    assert Settings(serving_predict_url=url).serving_predict_url == url


@pytest.mark.parametrize("url", ["ftp://serving/predict", "serving:3001/predict", "/predict"])
def test_serving_predict_url_rejects_non_http_schemes(url: str) -> None:
    with pytest.raises(ValidationError, match="http"):
        Settings(serving_predict_url=url)
