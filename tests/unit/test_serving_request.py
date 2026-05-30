import pytest
from pydantic import ValidationError

from fraud.serving.service import PredictRequest


def test_request_accepts_valid_transaction() -> None:
    request = PredictRequest(card_id="1_2_3_5", amount=99.5, transaction_id="t-1")
    assert request.card_id == "1_2_3_5"
    assert request.amount == 99.5


def test_request_defaults_transaction_id_to_none() -> None:
    request = PredictRequest(card_id="1_2_3_5", amount=99.5)
    assert request.transaction_id is None


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
