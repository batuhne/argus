"""Configuration for the serving service, assembled from settings and params."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fraud.config import get_settings
from fraud.model_loader import ChampionLoadConfig
from fraud.params import load_params
from fraud.paths import FEATURE_REPO_DIR, FEATURE_SERVICE


@dataclass(frozen=True, slots=True)
class ServingConfig:
    tracking_uri: str
    model_name: str
    champion_alias: str
    feature_service: str
    feast_repo_dir: Path
    redis_host: str
    redis_port: int
    request_timeout_seconds: int
    load_deadline_seconds: float
    reload_interval_seconds: float

    @classmethod
    def from_settings(cls) -> ServingConfig:
        settings = get_settings()
        params = load_params()
        return cls(
            tracking_uri=settings.mlflow_tracking_uri,
            model_name=settings.argus_model_name,
            champion_alias=params.evaluation.champion_alias,
            feature_service=FEATURE_SERVICE,
            feast_repo_dir=FEATURE_REPO_DIR,
            redis_host=settings.redis_host,
            redis_port=settings.redis_port,
            request_timeout_seconds=settings.mlflow_request_timeout_seconds,
            load_deadline_seconds=settings.model_load_deadline_seconds,
            reload_interval_seconds=settings.champion_reload_interval_seconds,
        )

    @property
    def redis_connection(self) -> str:
        return f"{self.redis_host}:{self.redis_port}"

    @property
    def champion_load_config(self) -> ChampionLoadConfig:
        return ChampionLoadConfig(
            tracking_uri=self.tracking_uri,
            model_name=self.model_name,
            champion_alias=self.champion_alias,
            request_timeout_seconds=self.request_timeout_seconds,
            load_deadline_seconds=self.load_deadline_seconds,
        )
