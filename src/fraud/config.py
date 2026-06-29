"""Runtime settings loaded from the environment and .env via pydantic-settings."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration read from the environment and a local .env file.

    Connection defaults point at the local docker compose stack. Secrets are
    held as SecretStr so they do not leak through logs or repr.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "local"
    log_level: str = "INFO"
    log_json: bool = False
    seed: int = 42

    mlflow_tracking_uri: str = "http://localhost:5500"
    mlflow_experiment_name: str = "argus"
    argus_model_name: str = "argus_fraud_classifier"
    # Caps a single tracking call so serving startup can't hang on a slow MLflow.
    mlflow_request_timeout_seconds: int = 15
    # How long serving keeps retrying the champion load before it gives up and exits.
    model_load_deadline_seconds: float = 60.0
    # All three broker ports, so a host client still connects when one broker is down.
    kafka_bootstrap_servers: str = "localhost:19092,localhost:29092,localhost:39092"
    serving_predict_url: str = "http://localhost:3001/predict"
    # When set, /predict requires this bearer token; unset leaves the endpoint open for local dev.
    serving_api_key: SecretStr | None = None
    monitoring_exporter_port: int = 8000
    redis_host: str = "localhost"
    redis_port: int = 6379

    kaggle_api_token: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
