import pytest
from pydantic import ValidationError

from fraud.ingestion.stream import (
    DriftAlertEvent,
    LabelEvent,
    PredictionEvent,
    RawAttributes,
    ScoredFeaturesEvent,
    TransactionEvent,
    deserialize_label,
    deserialize_scored_features,
    deserialize_transaction,
    seconds_per_message,
    serialize,
)
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
    assert tuple(RawAttributes.model_fields) == fl.RAW_NUMERIC_PASSTHROUGH


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


def test_seconds_per_message_inverts_rate() -> None:
    assert seconds_per_message(50.0) == pytest.approx(0.02)


@pytest.mark.parametrize("rate", [0.0, -5.0])
def test_seconds_per_message_rejects_non_positive_rate(rate: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        seconds_per_message(rate)
