"""Stream event models and the serving predict response contract.

Plus topic and group name constants, serialization, and payload bounds.
"""

from __future__ import annotations

from typing import Annotated, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StringConstraints

OptionalFiniteFloat: TypeAlias = FiniteFloat | None

# Abuse bounds on the request contract: a request past these is malformed, not a sale.
MAX_TRANSACTION_AMOUNT = 1_000_000.0
MAX_RAW_VECTOR_ENTRIES = 256
MAX_CATEGORICAL_VALUE_LENGTH = 256

SECONDS_PER_DAY = 86400.0

TRANSACTIONS_TOPIC = "transactions"
PREDICTIONS_TOPIC = "predictions"
SCORED_FEATURES_TOPIC = "scored-features"
LABELS_TOPIC = "labels"
DRIFT_ALERTS_TOPIC = "drift-alerts"
DLQ_TOPIC = "transactions-dlq"
CONSUMER_GROUP = "argus-fraud-consumer"
MONITOR_GROUP = "argus-fraud-monitor"
RETRAIN_GROUP = "argus-fraud-retrainer"


class RawAttributes(BaseModel):
    """Optional per-transaction numerics on the request; mirrors RAW_NUMERIC_PASSTHROUGH."""

    model_config = ConfigDict(extra="forbid")  # a misnamed field is an error, not a silent NaN

    C1: OptionalFiniteFloat = None
    C2: OptionalFiniteFloat = None
    C3: OptionalFiniteFloat = None
    C4: OptionalFiniteFloat = None
    C5: OptionalFiniteFloat = None
    C6: OptionalFiniteFloat = None
    C7: OptionalFiniteFloat = None
    C8: OptionalFiniteFloat = None
    C9: OptionalFiniteFloat = None
    C10: OptionalFiniteFloat = None
    C11: OptionalFiniteFloat = None
    C12: OptionalFiniteFloat = None
    C13: OptionalFiniteFloat = None
    C14: OptionalFiniteFloat = None
    D1: OptionalFiniteFloat = None
    D2: OptionalFiniteFloat = None
    D3: OptionalFiniteFloat = None
    D4: OptionalFiniteFloat = None
    D5: OptionalFiniteFloat = None
    D6: OptionalFiniteFloat = None
    D7: OptionalFiniteFloat = None
    D8: OptionalFiniteFloat = None
    D9: OptionalFiniteFloat = None
    D10: OptionalFiniteFloat = None
    D11: OptionalFiniteFloat = None
    D12: OptionalFiniteFloat = None
    D13: OptionalFiniteFloat = None
    D14: OptionalFiniteFloat = None
    D15: OptionalFiniteFloat = None
    dist1: OptionalFiniteFloat = None
    dist2: OptionalFiniteFloat = None
    addr1: OptionalFiniteFloat = None
    addr2: OptionalFiniteFloat = None
    # The reduced V set is frozen at selection time, so it rides as a dict keyed by name
    # rather than fixed fields; serving reindexes it to feature_logic.V_SELECTED.
    v: dict[str, OptionalFiniteFloat] = Field(
        default_factory=dict, max_length=MAX_RAW_VECTOR_ENTRIES
    )
    # Curated categoricals ride the same way; serving reindexes them to CATEGORICAL_COLUMNS
    # and the encoder turns them into features.
    categorical: dict[
        str, Annotated[str, StringConstraints(max_length=MAX_CATEGORICAL_VALUE_LENGTH)] | None
    ] = Field(default_factory=dict, max_length=MAX_RAW_VECTOR_ENTRIES)


class TransactionEvent(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=128)
    card_id: str = Field(min_length=1, max_length=128)
    amount: float = Field(gt=0.0, le=MAX_TRANSACTION_AMOUNT)
    event_timestamp: str
    raw: RawAttributes = Field(default_factory=RawAttributes)


class PredictResponse(BaseModel):
    """Serving's /predict reply, validated by the consumer into a shared contract.

    transaction_id and latency_ms are serving annotations the consumer ignores; the required
    core is the scoring fields it maps onto PredictionEvent.
    """

    transaction_id: str | None = Field(default=None, max_length=128)
    fraud_score: float = Field(ge=0.0, le=1.0)
    decision: bool
    threshold: float = Field(ge=0.0, le=1.0)
    model_version: int = Field(ge=1)
    latency_ms: float | None = Field(default=None, ge=0.0)


class PredictionEvent(BaseModel):
    transaction_id: str
    card_id: str
    fraud_score: float = Field(ge=0.0, le=1.0)
    decision: bool
    threshold: float = Field(ge=0.0, le=1.0)
    model_version: int = Field(ge=1)


class ScoredFeaturesEvent(BaseModel):
    """Inference log emitted by serving: the exact features the model scored."""

    transaction_id: str = Field(min_length=1, max_length=128)
    model_version: int = Field(ge=1)
    fraud_score: float = Field(ge=0.0, le=1.0)
    decision: bool
    features: dict[str, float | None]  # null marks a missing raw numeric


class LabelEvent(BaseModel):
    """Ground-truth outcome that arrives after the prediction (chargeback lag)."""

    transaction_id: str = Field(min_length=1, max_length=128)
    is_fraud: int = Field(ge=0, le=1)


class DriftAlertEvent(BaseModel):
    """Machine-readable retraining trigger published when a monitor breach persists."""

    kind: str = Field(min_length=1, max_length=64)
    metric: str = Field(min_length=1, max_length=64)
    value: FiniteFloat
    threshold: FiniteFloat
    detected_at: str = Field(min_length=1, max_length=64)


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
