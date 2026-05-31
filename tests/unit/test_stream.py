import pytest
from pydantic import ValidationError

from fraud.ingestion.stream import (
    PredictionEvent,
    TransactionEvent,
    deserialize_transaction,
    seconds_per_message,
    serialize,
)


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


def test_seconds_per_message_inverts_rate() -> None:
    assert seconds_per_message(50.0) == pytest.approx(0.02)


@pytest.mark.parametrize("rate", [0.0, -5.0])
def test_seconds_per_message_rejects_non_positive_rate(rate: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        seconds_per_message(rate)
