from typing import Any

import pytest
import requests

from fraud.ingestion.consumer import (
    MAX_PREDICT_RETRIES,
    _fetch_prediction,
    _ShutdownFlag,
    predict_request_body,
    prediction_from_response,
)
from fraud.ingestion.stream import StreamConfig, TransactionEvent


def _cfg() -> StreamConfig:
    return StreamConfig(
        bootstrap_servers="localhost:19092",
        transactions_topic="transactions",
        predictions_topic="predictions",
        consumer_group="argus-fraud-consumer",
        replay_rate_per_second=50.0,
        predict_url="http://localhost:3000/predict",
    )


def _event() -> TransactionEvent:
    return TransactionEvent(
        transaction_id="t-1",
        card_id="1_2_3_5",
        amount=12.5,
        event_timestamp="2017-12-01T00:00:00+00:00",
    )


_PREDICT_BODY = {"fraud_score": 0.12, "decision": False, "threshold": 0.07, "model_version": 5}


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse | Exception]) -> None:
        self._responses = responses
        self.calls = 0

    def post(self, url: str, json: Any, timeout: float) -> _FakeResponse:
        response = self._responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fraud.ingestion.consumer.time.sleep", lambda _seconds: None)


def test_predict_request_body_wraps_in_request_key() -> None:
    body = predict_request_body(_event())
    assert body == {"request": {"card_id": "1_2_3_5", "amount": 12.5, "transaction_id": "t-1"}}


def test_prediction_from_response_maps_fields() -> None:
    prediction = prediction_from_response(_event(), _PREDICT_BODY)
    assert prediction.transaction_id == "t-1"
    assert prediction.decision is False
    assert prediction.model_version == 5


def test_fetch_prediction_returns_event_on_success() -> None:
    session = _FakeSession([_FakeResponse(200, _PREDICT_BODY)])
    result = _fetch_prediction(_cfg(), session, _event(), _ShutdownFlag())  # type: ignore[arg-type]
    assert result is not None
    assert result.fraud_score == pytest.approx(0.12)
    assert session.calls == 1


def test_fetch_prediction_retries_then_succeeds_on_transient_error() -> None:
    session = _FakeSession([_FakeResponse(503), _FakeResponse(200, _PREDICT_BODY)])
    result = _fetch_prediction(_cfg(), session, _event(), _ShutdownFlag())  # type: ignore[arg-type]
    assert result is not None
    assert session.calls == 2


def test_fetch_prediction_retries_on_connection_error() -> None:
    session = _FakeSession([requests.ConnectionError("boom"), _FakeResponse(200, _PREDICT_BODY)])
    result = _fetch_prediction(_cfg(), session, _event(), _ShutdownFlag())  # type: ignore[arg-type]
    assert result is not None
    assert session.calls == 2


def test_fetch_prediction_skips_on_client_rejection() -> None:
    session = _FakeSession([_FakeResponse(422)])
    result = _fetch_prediction(_cfg(), session, _event(), _ShutdownFlag())  # type: ignore[arg-type]
    assert result is None
    assert session.calls == 1


def test_fetch_prediction_gives_up_after_max_retries() -> None:
    session = _FakeSession([_FakeResponse(503)] * MAX_PREDICT_RETRIES)
    result = _fetch_prediction(_cfg(), session, _event(), _ShutdownFlag())  # type: ignore[arg-type]
    assert result is None
    assert session.calls == MAX_PREDICT_RETRIES


def test_fetch_prediction_stops_when_shutdown_requested() -> None:
    shutdown = _ShutdownFlag()
    shutdown.requested = True
    session = _FakeSession([_FakeResponse(200, _PREDICT_BODY)])
    result = _fetch_prediction(_cfg(), session, _event(), shutdown)  # type: ignore[arg-type]
    assert result is None
    assert session.calls == 0
