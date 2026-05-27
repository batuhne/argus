"""Open the Feast store with the Redis endpoint taken from configuration."""

from __future__ import annotations

import os
from pathlib import Path

from feast import FeatureStore

from fraud.config import get_settings
from fraud.paths import FEATURE_REPO_DIR

# feature_store.yaml reads the online connection string from this variable.
REDIS_CONNECTION_ENV = "ARGUS_REDIS_CONNECTION"


def open_feature_store(repo_dir: Path = FEATURE_REPO_DIR) -> FeatureStore:
    settings = get_settings()
    os.environ.setdefault(REDIS_CONNECTION_ENV, f"{settings.redis_host}:{settings.redis_port}")
    return FeatureStore(repo_path=str(repo_dir))
