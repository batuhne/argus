from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from fraud.config import get_settings

SECONDS_PER_DAY = 86400.0

TRANSACTIONS_TOPIC = "transactions"
PREDICTIONS_TOPIC = "predictions"
SCORED_FEATURES_TOPIC = "scored-features"
LABELS_TOPIC = "labels"
DRIFT_ALERTS_TOPIC = "drift-alerts"
CONSUMER_GROUP = "argus-fraud-consumer"
MONITOR_GROUP = "argus-fraud-monitor"
RETRAIN_GROUP = "argus-fraud-retrainer"


@dataclass(frozen=True, slots=True)
class StreamConfig:
    bootstrap_servers: str
    transactions_topic: str
    predictions_topic: str
    consumer_group: str
    predict_url: str

    @classmethod
    def from_settings(cls) -> StreamConfig:
        settings = get_settings()
        return cls(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            transactions_topic=TRANSACTIONS_TOPIC,
            predictions_topic=PREDICTIONS_TOPIC,
            consumer_group=CONSUMER_GROUP,
            predict_url=settings.serving_predict_url,
        )


class RawAttributes(BaseModel):
    """Optional per-transaction numerics on the request; mirrors RAW_NUMERIC_PASSTHROUGH."""

    model_config = ConfigDict(extra="forbid")  # a misnamed field is an error, not a silent NaN

    C1: float | None = None
    C2: float | None = None
    C3: float | None = None
    C4: float | None = None
    C5: float | None = None
    C6: float | None = None
    C7: float | None = None
    C8: float | None = None
    C9: float | None = None
    C10: float | None = None
    C11: float | None = None
    C12: float | None = None
    C13: float | None = None
    C14: float | None = None
    D1: float | None = None
    D2: float | None = None
    D3: float | None = None
    D4: float | None = None
    D5: float | None = None
    D6: float | None = None
    D7: float | None = None
    D8: float | None = None
    D9: float | None = None
    D10: float | None = None
    D11: float | None = None
    D12: float | None = None
    D13: float | None = None
    D14: float | None = None
    D15: float | None = None
    dist1: float | None = None
    dist2: float | None = None
    addr1: float | None = None
    addr2: float | None = None
    # The reduced V set is frozen at selection time, so it rides as a dict keyed by name
    # rather than fixed fields; serving reindexes it to feature_logic.V_SELECTED.
    v: dict[str, float | None] = Field(default_factory=dict)
    # Curated categoricals ride the same way; serving reindexes them to CATEGORICAL_COLUMNS
    # and the encoder turns them into features.
    categorical: dict[str, str | None] = Field(default_factory=dict)


class TransactionEvent(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=128)
    card_id: str = Field(min_length=1, max_length=128)
    amount: float = Field(gt=0.0)
    event_timestamp: str
    raw: RawAttributes = Field(default_factory=RawAttributes)


class PredictionEvent(BaseModel):
    transaction_id: str
    card_id: str
    fraud_score: float
    decision: bool
    threshold: float
    model_version: int


class ScoredFeaturesEvent(BaseModel):
    """Inference log emitted by serving: the exact features the model scored."""

    transaction_id: str = Field(min_length=1, max_length=128)
    model_version: int
    fraud_score: float
    decision: bool
    features: dict[str, float | None]  # null marks a missing raw numeric


class LabelEvent(BaseModel):
    """Ground-truth outcome that arrives after the prediction (chargeback lag)."""

    transaction_id: str = Field(min_length=1, max_length=128)
    is_fraud: int = Field(ge=0, le=1)


class DriftAlertEvent(BaseModel):
    """Machine-readable retraining trigger published when a monitor breach persists."""

    kind: str
    metric: str
    value: float
    threshold: float
    detected_at: str


def serialize(event: BaseModel) -> bytes:
    return event.model_dump_json().encode("utf-8")


def deserialize_transaction(payload: bytes) -> TransactionEvent:
    return TransactionEvent.model_validate_json(payload)


def deserialize_scored_features(payload: bytes) -> ScoredFeaturesEvent:
    return ScoredFeaturesEvent.model_validate_json(payload)


def deserialize_label(payload: bytes) -> LabelEvent:
    return LabelEvent.model_validate_json(payload)


def deserialize_drift_alert(payload: bytes) -> DriftAlertEvent:
    return DriftAlertEvent.model_validate_json(payload)


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
