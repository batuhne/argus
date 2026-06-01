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
    kafka_bootstrap_servers: str = "localhost:19092"
    serving_predict_url: str = "http://localhost:3001/predict"
    stream_replay_rate: float = 50.0
    stream_label_delay_seconds: float = 5.0
    monitoring_exporter_port: int = 8000
    redis_host: str = "localhost"
    redis_port: int = 6379

    kaggle_api_token: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
