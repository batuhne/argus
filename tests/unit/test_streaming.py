import pandas as pd
import pytest
from pydantic import ValidationError

from fraud.streaming.events import (
    MAX_CATEGORICAL_VALUE_LENGTH,
    MAX_RAW_VECTOR_ENTRIES,
    MAX_TRANSACTION_AMOUNT,
    DriftAlertEvent,
    LabelEvent,
    PredictionEvent,
    RawAttributes,
    ScoredFeaturesEvent,
    TransactionEvent,
    deserialize_label,
    deserialize_scored_features,
    deserialize_transaction,
    serialize,
)
from fraud.streaming.transport import durable_producer_config, replay_step_delays
from fraud.transforms import feature_logic as fl


def _transaction() -> TransactionEvent:
    return TransactionEvent(
        transaction_id="t-1",
        card_id="1_2_3_5",
        amount=12.5,
        event_timestamp="2017-12-01T00:00:00+00:00",
    )


def test_transaction_event_survives_json_round_trip() -> None:
    restored = deserialize_transaction(serialize(_transaction()))
    assert restored == _transaction()


def test_raw_attributes_fields_match_passthrough_contract() -> None:
    dict_fields = {"v", "categorical"}
    numeric_fields = tuple(name for name in RawAttributes.model_fields if name not in dict_fields)
    assert numeric_fields == fl.RAW_NUMERIC_PASSTHROUGH


def test_transaction_event_carries_v_vector_round_trip() -> None:
    event = TransactionEvent(
        transaction_id="t-1",
        card_id="1_2_3_5",
        amount=12.5,
        event_timestamp="2017-12-01T00:00:00+00:00",
        raw=RawAttributes(v={"V147": 0.5, "V201": None}),
    )
    restored = deserialize_transaction(serialize(event))
    assert restored.raw.v == {"V147": 0.5, "V201": None}


def test_transaction_event_carries_categorical_vector_round_trip() -> None:
    event = TransactionEvent(
        transaction_id="t-1",
        card_id="1_2_3_5",
        amount=12.5,
        event_timestamp="2017-12-01T00:00:00+00:00",
        raw=RawAttributes(categorical={"ProductCD": "W", "DeviceType": None}),
    )
    restored = deserialize_transaction(serialize(event))
    assert restored.raw.categorical == {"ProductCD": "W", "DeviceType": None}


def test_raw_attributes_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        RawAttributes.model_validate({"c1": 5.0})


def test_transaction_event_carries_raw_attributes_round_trip() -> None:
    event = TransactionEvent(
        transaction_id="t-1",
        card_id="1_2_3_5",
        amount=12.5,
        event_timestamp="2017-12-01T00:00:00+00:00",
        raw=RawAttributes(C1=3.0, dist1=1.5),
    )
    restored = deserialize_transaction(serialize(event))
    assert restored.raw.C1 == 3.0
    assert restored.raw.dist1 == 1.5
    assert restored.raw.C2 is None


def test_prediction_event_serializes_to_json_bytes() -> None:
    event = PredictionEvent(
        transaction_id="t-1",
        card_id="1_2_3_5",
        fraud_score=0.12,
        decision=False,
        threshold=0.07,
        model_version=5,
    )
    assert serialize(event).startswith(b"{")


def test_transaction_event_rejects_non_positive_amount() -> None:
    with pytest.raises(ValidationError):
        TransactionEvent(
            transaction_id="t-1", card_id="c", amount=0.0, event_timestamp="2017-12-01T00:00:00"
        )


def test_transaction_event_rejects_empty_card_id() -> None:
    with pytest.raises(ValidationError):
        TransactionEvent(
            transaction_id="t-1", card_id="", amount=1.0, event_timestamp="2017-12-01T00:00:00"
        )


def test_transaction_event_rejects_amount_above_cap() -> None:
    with pytest.raises(ValidationError):
        TransactionEvent(
            transaction_id="t-1",
            card_id="c",
            amount=MAX_TRANSACTION_AMOUNT + 1.0,
            event_timestamp="2017-12-01T00:00:00",
        )


def test_raw_attributes_rejects_oversized_categorical_value() -> None:
    with pytest.raises(ValidationError):
        RawAttributes(categorical={"ProductCD": "W" * (MAX_CATEGORICAL_VALUE_LENGTH + 1)})


def test_raw_attributes_rejects_too_many_vector_entries() -> None:
    with pytest.raises(ValidationError):
        RawAttributes(v={f"V{i}": 1.0 for i in range(MAX_RAW_VECTOR_ENTRIES + 1)})


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_raw_attributes_rejects_non_finite_numeric(value: float) -> None:
    with pytest.raises(ValidationError):
        RawAttributes(C1=value)


def test_raw_attributes_rejects_non_finite_v_value() -> None:
    with pytest.raises(ValidationError):
        RawAttributes(v={"V147": float("inf")})


def test_raw_attributes_rejects_non_finite_json_token() -> None:
    # pydantic's JSON parser accepts the Infinity token; FiniteFloat is what rejects the value.
    with pytest.raises(ValidationError):
        RawAttributes.model_validate_json('{"C1": Infinity}')


def test_deserialize_rejects_malformed_payload() -> None:
    with pytest.raises(ValidationError):
        deserialize_transaction(b'{"transaction_id": "t-1"}')


def _scored_features() -> ScoredFeaturesEvent:
    return ScoredFeaturesEvent(
        transaction_id="t-1",
        model_version=5,
        fraud_score=0.42,
        decision=True,
        features={"amt_log": 3.1, "TransactionAmt": 21.0},
    )


def test_scored_features_event_survives_json_round_trip() -> None:
    restored = deserialize_scored_features(serialize(_scored_features()))
    assert restored == _scored_features()


def test_scored_features_event_round_trips_missing_feature_as_null() -> None:
    event = ScoredFeaturesEvent(
        transaction_id="t-1",
        model_version=6,
        fraud_score=0.5,
        decision=True,
        features={"C1": None, "TransactionAmt": 21.0},
    )
    restored = deserialize_scored_features(serialize(event))
    assert restored.features["C1"] is None
    assert restored.features["TransactionAmt"] == 21.0


def test_scored_features_event_rejects_empty_transaction_id() -> None:
    with pytest.raises(ValidationError):
        ScoredFeaturesEvent(
            transaction_id="", model_version=1, fraud_score=0.1, decision=False, features={}
        )


def test_label_event_round_trip_and_range() -> None:
    restored = deserialize_label(serialize(LabelEvent(transaction_id="t-1", is_fraud=1)))
    assert restored.is_fraud == 1


@pytest.mark.parametrize("value", [-1, 2])
def test_label_event_rejects_out_of_range(value: int) -> None:
    with pytest.raises(ValidationError):
        LabelEvent(transaction_id="t-1", is_fraud=value)


def test_drift_alert_event_serializes() -> None:
    event = DriftAlertEvent(
        kind="data_drift",
        metric="feature_drift_psi",
        value=0.31,
        threshold=0.2,
        detected_at="2017-12-01T00:00:00+00:00",
    )
    assert serialize(event).startswith(b"{")


def test_replay_step_delays_warps_real_gaps_with_zero_first_wait() -> None:
    dt = pd.Series([100, 110, 140])
    delays = replay_step_delays(dt, time_warp_factor=10.0, max_step_seconds=100.0)
    # First message has no predecessor; gaps 10s and 30s warped by 10 give 1s and 3s.
    assert delays.tolist() == pytest.approx([0.0, 1.0, 3.0])


def test_replay_step_delays_caps_long_gaps() -> None:
    dt = pd.Series([0, 10, 1_000_000])
    delays = replay_step_delays(dt, time_warp_factor=1.0, max_step_seconds=5.0)
    assert delays.tolist() == pytest.approx([0.0, 5.0, 5.0])


@pytest.mark.parametrize("warp", [0.0, -1.0])
def test_replay_step_delays_rejects_non_positive_warp(warp: float) -> None:
    with pytest.raises(ValueError, match="time_warp_factor must be positive"):
        replay_step_delays(pd.Series([0, 1]), time_warp_factor=warp, max_step_seconds=1.0)


def test_durable_producer_config_enables_idempotence() -> None:
    cfg = durable_producer_config("redpanda-0:9092,redpanda-1:9092")
    assert cfg["bootstrap.servers"] == "redpanda-0:9092,redpanda-1:9092"
    # The flag that buys durability and retry dedup.
    assert cfg["enable.idempotence"] == "true"
