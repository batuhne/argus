from fraud.config import Settings, get_settings


def test_settings_has_expected_shape() -> None:
    settings = Settings()
    assert settings.seed == 42
    assert isinstance(settings.redis_port, int)
    assert settings.mlflow_tracking_uri.startswith("http")


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_bootstrap_default_lists_every_broker() -> None:
    # The code default, not a .env override.
    default = Settings.model_fields["kafka_bootstrap_servers"].default
    brokers = default.split(",")
    assert len(brokers) == 3
    assert all(":" in broker for broker in brokers)
