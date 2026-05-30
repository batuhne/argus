from __future__ import annotations

import os
import socket

import pandas as pd
from feast import FeatureStore

from fraud.serving.config import ServingConfig
from fraud.training.features import FEATURE_COLUMNS

REDIS_PROBE_TIMEOUT_SECONDS = 1.0


class OnlineFeatureFetcher:
    """Reads card_activity features from Feast online (Redis) for one transaction."""

    def __init__(self, cfg: ServingConfig) -> None:
        os.environ.setdefault("ARGUS_REDIS_CONNECTION", cfg.redis_connection)
        self._store = FeatureStore(repo_path=str(cfg.feast_repo_dir))
        self._service = self._store.get_feature_service(cfg.feature_service)

    def fetch(self, card_id: str, amount: float) -> pd.DataFrame:
        response = self._store.get_online_features(
            features=self._service,
            entity_rows=[{"card_id": card_id, "TransactionAmt": amount}],
        )
        return assemble_features(response.to_df(), amount)


def assemble_features(online_frame: pd.DataFrame, amount: float) -> pd.DataFrame:
    """Order the online feature row to the model's columns, adding the raw amount."""
    frame = online_frame.copy()
    frame["TransactionAmt"] = amount
    return frame.loc[:, list(FEATURE_COLUMNS)]


def redis_reachable(cfg: ServingConfig) -> bool:
    try:
        with socket.create_connection(
            (cfg.redis_host, cfg.redis_port), timeout=REDIS_PROBE_TIMEOUT_SECONDS
        ):
            return True
    except OSError:
        return False
