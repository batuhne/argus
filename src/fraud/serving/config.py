from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fraud.config import get_settings
from fraud.paths import FEATURE_REPO_DIR

FEATURE_SERVICE = "card_activity"


@dataclass(frozen=True, slots=True)
class ServingConfig:
    tracking_uri: str
    model_name: str
    champion_alias: str
    feature_service: str
    feast_repo_dir: Path
    redis_host: str
    redis_port: int

    @classmethod
    def from_settings(cls) -> ServingConfig:
        settings = get_settings()
        return cls(
            tracking_uri=settings.mlflow_tracking_uri,
            model_name=settings.argus_model_name,
            champion_alias="champion",
            feature_service=FEATURE_SERVICE,
            feast_repo_dir=FEATURE_REPO_DIR,
            redis_host=settings.redis_host,
            redis_port=settings.redis_port,
        )

    @property
    def redis_connection(self) -> str:
        return f"{self.redis_host}:{self.redis_port}"
