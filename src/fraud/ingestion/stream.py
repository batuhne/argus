from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from fraud.config import get_settings

TRANSACTIONS_TOPIC = "transactions"
PREDICTIONS_TOPIC = "predictions"
SCORED_FEATURES_TOPIC = "scored-features"
LABELS_TOPIC = "labels"
DRIFT_ALERTS_TOPIC = "drift-alerts"
CONSUMER_GROUP = "argus-fraud-consumer"
MONITOR_GROUP = "argus-fraud-monitor"


@dataclass(frozen=True, slots=True)
class StreamConfig:
    bootstrap_servers: str
    transactions_topic: str
    predictions_topic: str
    consumer_group: str
    replay_rate_per_second: float
    predict_url: str

    @classmethod
    def from_settings(cls) -> StreamConfig:
        settings = get_settings()
        return cls(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            transactions_topic=TRANSACTIONS_TOPIC,
            predictions_topic=PREDICTIONS_TOPIC,
            consumer_group=CONSUMER_GROUP,
            replay_rate_per_second=settings.stream_replay_rate,
            predict_url=settings.serving_predict_url,
        )


class TransactionEvent(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=128)
    card_id: str = Field(min_length=1, max_length=128)
    amount: float = Field(gt=0.0)
    event_timestamp: str


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
    features: dict[str, float]


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


def seconds_per_message(rate_per_second: float) -> float:
    """Inter-message delay that paces the replay to the target throughput."""
    if rate_per_second <= 0.0:
        raise ValueError(f"replay rate must be positive, got {rate_per_second}")
    return 1.0 / rate_per_second
