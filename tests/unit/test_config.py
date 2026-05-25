from fraud.config import Settings, get_settings


def test_settings_has_expected_shape() -> None:
    settings = Settings()
    assert settings.seed == 42
    assert isinstance(settings.redis_port, int)
    assert settings.mlflow_tracking_uri.startswith("http")


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
