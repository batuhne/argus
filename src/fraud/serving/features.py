from __future__ import annotations

import os
import socket

import pandas as pd
from feast import FeatureStore

from fraud.ingestion.stream import RawAttributes
from fraud.serving.config import ServingConfig
from fraud.training.features import FEATURE_COLUMNS
from fraud.transforms import feature_logic as fl
from fraud.transforms.encoders import CategoricalEncoder

REDIS_PROBE_TIMEOUT_SECONDS = 1.0


class OnlineFeatureFetcher:
    """Reads card_activity features from Feast online (Redis) for one transaction."""

    def __init__(self, cfg: ServingConfig, encoder: CategoricalEncoder) -> None:
        os.environ.setdefault("ARGUS_REDIS_CONNECTION", cfg.redis_connection)
        self._store = FeatureStore(repo_path=str(cfg.feast_repo_dir))
        self._service = self._store.get_feature_service(cfg.feature_service)
        self._encoder = encoder

    def fetch(self, card_id: str, amount: float, raw: RawAttributes) -> pd.DataFrame:
        response = self._store.get_online_features(
            features=self._service,
            entity_rows=[{"card_id": card_id, "TransactionAmt": amount}],
        )
        return assemble_features(response.to_df(), amount, raw, self._encoder)


def assemble_features(
    online_frame: pd.DataFrame, amount: float, raw: RawAttributes, encoder: CategoricalEncoder
) -> pd.DataFrame:
    """Assemble the online row, raw numerics, and encoded categoricals into the model contract."""
    frame = online_frame.copy()
    frame["TransactionAmt"] = amount
    payload = raw.model_dump()
    v_values = payload.pop("v")
    categorical_values = payload.pop("categorical")
    raw_frame = pd.DataFrame([payload], index=frame.index)
    # Drop any V the request sent outside the frozen set and fill any it omitted with NaN.
    v_frame = pd.DataFrame([v_values], index=frame.index).reindex(columns=list(fl.V_SELECTED))
    # Same for categoricals; the encoder maps an omitted column to its missing-category value.
    categorical_frame = pd.DataFrame([categorical_values], index=frame.index).reindex(
        columns=list(fl.CATEGORICAL_COLUMNS)
    )
    encoded = encoder.transform(categorical_frame)
    frame = pd.concat([frame, raw_frame, v_frame, encoded], axis=1)
    frame = fl.coerce_numeric(frame, [*fl.RAW_NUMERIC_PASSTHROUGH, *fl.V_SELECTED])
    return frame.loc[:, list(FEATURE_COLUMNS)]


def redis_reachable(cfg: ServingConfig) -> bool:
    try:
        with socket.create_connection(
            (cfg.redis_host, cfg.redis_port), timeout=REDIS_PROBE_TIMEOUT_SECONDS
        ):
            return True
    except OSError:
        return False
