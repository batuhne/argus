"""Assemble the model feature row from the request and Feast; also the Redis readiness probe."""

from __future__ import annotations

import os

import pandas as pd
import redis
from feast import FeatureStore
from prometheus_client import Counter, Histogram

from fraud.serving.config import ServingConfig
from fraud.streaming.events import RawAttributes
from fraud.transforms import feature_logic as fl
from fraud.transforms.encoders import CategoricalEncoder
from fraud.transforms.features import FEATURE_COLUMNS

REDIS_PROBE_TIMEOUT_SECONDS = 1.0

FEATURE_FETCH_SECONDS = Histogram(
    "argus_feature_fetch_seconds", "Latency of the online feature-store read"
)
FEATURE_FETCH_ERRORS = Counter(
    "argus_feature_fetch_errors_total", "Online feature-store read failures"
)


class OnlineFeatureFetcher:
    """Reads card_activity features from Feast online (Redis) for one transaction."""

    def __init__(self, cfg: ServingConfig, encoder: CategoricalEncoder) -> None:
        os.environ.setdefault("ARGUS_REDIS_CONNECTION", cfg.redis_connection)
        self._store = FeatureStore(repo_path=str(cfg.feast_repo_dir))
        self._service = self._store.get_feature_service(cfg.feature_service)
        self._encoder = encoder

    def fetch(self, card_id: str, amount: float, raw: RawAttributes) -> pd.DataFrame:
        # Counted and timed separately so a slow Redis is distinguishable from a slow request.
        with FEATURE_FETCH_SECONDS.time():
            try:
                response = self._store.get_online_features(
                    features=self._service,
                    entity_rows=[{"card_id": card_id, "TransactionAmt": amount}],
                )
            except Exception:
                FEATURE_FETCH_ERRORS.inc()
                raise
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
    frame = fl.fill_missing_velocity(frame)
    return frame.loc[:, list(FEATURE_COLUMNS)]


def redis_reachable(cfg: ServingConfig) -> bool:
    # PING, not a bare connect: a port open but not answering commands is not ready.
    client = redis.Redis(
        host=cfg.redis_host,
        port=cfg.redis_port,
        socket_connect_timeout=REDIS_PROBE_TIMEOUT_SECONDS,
        socket_timeout=REDIS_PROBE_TIMEOUT_SECONDS,
    )
    try:
        return bool(client.ping())
    except (OSError, redis.RedisError):
        return False
    finally:
        client.close()
