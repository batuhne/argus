import pytest
from pydantic import ValidationError

from fraud.ingestion.stream import (
    MAX_CATEGORICAL_VALUE_LENGTH,
    MAX_RAW_VECTOR_ENTRIES,
    MAX_TRANSACTION_AMOUNT,
)
from fraud.serving.service import PredictRequest


def test_request_accepts_valid_transaction() -> None:
    request = PredictRequest(card_id="1_2_3_5", amount=99.5, transaction_id="t-1")
    assert request.card_id == "1_2_3_5"
    assert request.amount == 99.5


def test_request_defaults_transaction_id_to_none() -> None:
    request = PredictRequest(card_id="1_2_3_5", amount=99.5)
    assert request.transaction_id is None


def test_request_defaults_raw_to_empty_attributes() -> None:
    request = PredictRequest(card_id="1_2_3_5", amount=99.5)
    assert request.raw.C1 is None


def test_request_accepts_raw_attributes() -> None:
    request = PredictRequest.model_validate(
        {"card_id": "1_2_3_5", "amount": 99.5, "raw": {"C1": 5.0}}
    )
    assert request.raw.C1 == 5.0


def test_request_rejects_empty_card_id() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(card_id="", amount=99.5)


@pytest.mark.parametrize("amount", [0.0, -1.0])
def test_request_rejects_non_positive_amount(amount: float) -> None:
    with pytest.raises(ValidationError):
        PredictRequest(card_id="1_2_3_5", amount=amount)


def test_request_rejects_oversized_card_id() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(card_id="x" * 129, amount=99.5)


def test_request_accepts_amount_at_the_cap() -> None:
    request = PredictRequest(card_id="c", amount=MAX_TRANSACTION_AMOUNT)
    assert request.amount == MAX_TRANSACTION_AMOUNT


def test_request_rejects_amount_above_the_cap() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(card_id="c", amount=MAX_TRANSACTION_AMOUNT + 1.0)


def test_request_rejects_unknown_top_level_field() -> None:
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({"card_id": "c", "amount": 1.0, "rogue": 1})


def test_request_rejects_oversized_categorical_value() -> None:
    payload = {"card_id": "c", "amount": 1.0, "raw": {"categorical": {"ProductCD": "W" * 300}}}
    with pytest.raises(ValidationError):
        PredictRequest.model_validate(payload)


def test_request_rejects_too_many_raw_vector_entries() -> None:
    flooded = {f"V{i}": 1.0 for i in range(MAX_RAW_VECTOR_ENTRIES + 1)}
    payload = {"card_id": "c", "amount": 1.0, "raw": {"v": flooded}}
    with pytest.raises(ValidationError):
        PredictRequest.model_validate(payload)


def test_request_accepts_categorical_value_at_the_length_cap() -> None:
    payload = {
        "card_id": "c",
        "amount": 1.0,
        "raw": {"categorical": {"DeviceInfo": "d" * MAX_CATEGORICAL_VALUE_LENGTH}},
    }
    assert PredictRequest.model_validate(payload).raw.categorical["DeviceInfo"]
