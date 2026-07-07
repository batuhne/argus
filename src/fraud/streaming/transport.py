"""Stream transport: connection config, durable producer settings, and replay pacing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fraud.config import get_settings
from fraud.streaming.events import (
    CONSUMER_GROUP,
    DLQ_TOPIC,
    PREDICTIONS_TOPIC,
    TRANSACTIONS_TOPIC,
)


@dataclass(frozen=True, slots=True)
class StreamConfig:
    bootstrap_servers: str
    transactions_topic: str
    predictions_topic: str
    dlq_topic: str
    consumer_group: str
    predict_url: str
    # repr-suppressed so a traceback or log that captures the config never leaks the bearer token.
    predict_api_key: str | None = field(default=None, repr=False)
    metrics_port: int = 8001

    @classmethod
    def from_settings(cls) -> StreamConfig:
        settings = get_settings()
        api_key = settings.serving_api_key
        return cls(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            transactions_topic=TRANSACTIONS_TOPIC,
            predictions_topic=PREDICTIONS_TOPIC,
            dlq_topic=DLQ_TOPIC,
            consumer_group=CONSUMER_GROUP,
            predict_url=settings.serving_predict_url,
            predict_api_key=api_key.get_secret_value() if api_key is not None else None,
            metrics_port=settings.consumer_metrics_port,
        )


def durable_producer_config(bootstrap_servers: str) -> dict[str, str]:
    """Idempotent acks=all producer: a write survives one broker loss and retries do not
    duplicate. Not for serving's best-effort inference log."""
    return {"bootstrap.servers": bootstrap_servers, "enable.idempotence": "true"}


def replay_step_delays(
    transaction_dt: pd.Series, *, time_warp_factor: float, max_step_seconds: float
) -> NDArray[np.float64]:
    """Wall-clock wait before each message: ascending TransactionDT gaps compressed by the warp.

    First wait is 0; gaps are capped so a long real-world lull cannot stall the replay.
    """
    if time_warp_factor <= 0.0:
        raise ValueError(f"time_warp_factor must be positive, got {time_warp_factor}")
    steps = (transaction_dt.diff().fillna(0.0) / time_warp_factor).clip(0.0, max_step_seconds)
    return np.asarray(steps.to_numpy(), dtype=np.float64)
