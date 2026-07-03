import json
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from typing import Any

import pytest
import requests

from fraud.common.shutdown import ShutdownFlag
from fraud.ingestion.consumer import (
    BACKOFF_BASE_SECONDS,
    BACKOFF_MAX_SECONDS,
    MAX_PREDICT_RETRIES,
    FetchOutcome,
    _CircuitBreaker,
    _fetch_prediction,
    _handle_message,
    _interruptible_sleep,
    _retry_after_seconds,
    _seek_back,
    _shed_delay,
    predict_request_body,
    prediction_from_response,
    run_consumer,
)
from fraud.streaming.events import TransactionEvent, serialize
from fraud.streaming.transport import StreamConfig


def _cfg(predict_api_key: str | None = None) -> StreamConfig:
    return StreamConfig(
        bootstrap_servers="localhost:19092",
        transactions_topic="transactions",
        predictions_topic="predictions",
        dlq_topic="transactions-dlq",
        consumer_group="argus-fraud-consumer",
        predict_url="http://localhost:3000/predict",
        predict_api_key=predict_api_key,
    )


def _event() -> TransactionEvent:
    return TransactionEvent(
        transaction_id="t-1",
        card_id="1_2_3_5",
        amount=12.5,
        event_timestamp="2017-12-01T00:00:00+00:00",
    )


def _payload() -> bytes:
    return serialize(_event())


def _breaker() -> _CircuitBreaker:
    return _CircuitBreaker(threshold=3, cooldown_seconds=10.0)


# transaction_id and latency_ms omitted on purpose: optional, and the consumer ignores them.
_PREDICT_BODY = {"fraud_score": 0.12, "decision": False, "threshold": 0.07, "model_version": 5}


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    @property
    def content(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse | Exception]) -> None:
        self._responses = responses
        self.calls = 0
        self.headers: Any = None

    def post(self, url: str, json: Any, timeout: float, headers: Any = None) -> _FakeResponse:
        self.headers = headers
        response = self._responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        pass


class _FakeProducer:
    def __init__(self, *, deliver: bool = True) -> None:
        self.produced: list[tuple[str, bytes | None, Any]] = []
        self._deliver = deliver
        self._callbacks: list[Any] = []

    def produce(
        self,
        topic: str,
        value: Any = None,
        key: Any = None,
        headers: Any = None,
        on_delivery: Any = None,
    ) -> None:
        self.produced.append((topic, value, headers))
        if on_delivery is not None:
            self._callbacks.append(on_delivery)

    def flush(self, timeout: float | None = None) -> int:
        error = None if self._deliver else "broker unavailable"
        for callback in self._callbacks:
            callback(error, None)
        self._callbacks.clear()
        return 0 if self._deliver else 1


class _FakeConsumer:
    def __init__(self) -> None:
        self.commits = 0
        self.seeks: list[Any] = []

    def commit(self, message: Any, asynchronous: bool) -> None:
        self.commits += 1

    def seek(self, partition: Any) -> None:
        self.seeks.append(partition)


class _FakeMessage:
    def __init__(
        self,
        value: bytes | None,
        *,
        key: bytes | None = b"k",
        topic: str = "transactions",
        partition: int = 0,
        offset: int = 7,
    ) -> None:
        self._value = value
        self._key = key
        self._topic = topic
        self._partition = partition
        self._offset = offset

    def value(self) -> bytes | None:
        return self._value

    def key(self) -> bytes | None:
        return self._key

    def topic(self) -> str:
        return self._topic

    def partition(self) -> int:
        return self._partition

    def offset(self) -> int:
        return self._offset

    def error(self) -> None:
        return None


class _LoopConsumer:
    def __init__(self, message: Any, *, deliver_count: int, shutdown: ShutdownFlag) -> None:
        self._message = message
        self._remaining = deliver_count
        self._shutdown = shutdown
        self.commits = 0
        self.seeks: list[Any] = []

    def subscribe(self, topics: Any, **kwargs: Any) -> None:
        pass

    def poll(self, timeout: float) -> Any:
        if self._remaining <= 0:
            self._shutdown.requested = True
            return None
        self._remaining -= 1
        return self._message

    def commit(self, message: Any, asynchronous: bool) -> None:
        self.commits += 1

    def seek(self, partition: Any) -> None:
        self.seeks.append(partition)

    def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fraud.ingestion.consumer.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "fraud.ingestion.consumer._interruptible_sleep", lambda _seconds, _shutdown: None
    )


def test_predict_request_body_wraps_in_request_key() -> None:
    request = predict_request_body(_event())["request"]
    assert request["card_id"] == "1_2_3_5"
    assert request["amount"] == 12.5
    assert request["transaction_id"] == "t-1"


def test_predict_request_body_forwards_raw_vector() -> None:
    request = predict_request_body(_event())["request"]
    assert request["raw"]["C1"] is None


def test_prediction_from_response_maps_fields() -> None:
    prediction = prediction_from_response(_event(), _FakeResponse(200, _PREDICT_BODY))  # type: ignore[arg-type]
    assert prediction is not None
    assert prediction.transaction_id == "t-1"
    assert prediction.decision is False
    assert prediction.model_version == 5


def test_prediction_from_response_returns_none_on_malformed_body() -> None:
    # A 2xx whose body breaks the contract must not crash the loop on a missing field.
    response = _FakeResponse(200, {"fraud_score": 0.1})
    assert prediction_from_response(_event(), response) is None  # type: ignore[arg-type]


def test_fetch_prediction_unavailable_on_malformed_success_body() -> None:
    session = _FakeSession([_FakeResponse(200, {"fraud_score": 0.1})])
    result = _fetch_prediction(_cfg(), session, _event(), ShutdownFlag())  # type: ignore[arg-type]
    assert result.outcome is FetchOutcome.UNAVAILABLE
    assert session.calls == 1


def test_fetch_prediction_returns_event_on_success() -> None:
    session = _FakeSession([_FakeResponse(200, _PREDICT_BODY)])
    result = _fetch_prediction(_cfg(), session, _event(), ShutdownFlag())  # type: ignore[arg-type]
    assert result.outcome is FetchOutcome.OK
    assert result.prediction is not None
    assert result.prediction.fraud_score == pytest.approx(0.12)
    assert session.calls == 1


def test_fetch_prediction_retries_then_succeeds_on_transient_error() -> None:
    session = _FakeSession([_FakeResponse(503), _FakeResponse(200, _PREDICT_BODY)])
    result = _fetch_prediction(_cfg(), session, _event(), ShutdownFlag())  # type: ignore[arg-type]
    assert result.outcome is FetchOutcome.OK
    assert session.calls == 2


def test_fetch_prediction_yields_to_breaker_on_too_many_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []
    monkeypatch.setattr(
        "fraud.ingestion.consumer._interruptible_sleep",
        lambda seconds, _shutdown: delays.append(seconds),
    )
    session = _FakeSession([_FakeResponse(429, headers={"Retry-After": "2"})])
    result = _fetch_prediction(_cfg(), session, _event(), ShutdownFlag())  # type: ignore[arg-type]
    assert result.outcome is FetchOutcome.UNAVAILABLE
    assert session.calls == 1  # one shot, then yield to the breaker; no per-message retry storm
    assert delays == [2.0]  # honored Retry-After


def test_retry_after_seconds_parses_delta_seconds() -> None:
    response = _FakeResponse(429, headers={"Retry-After": "5"})
    assert _retry_after_seconds(response) == 5.0  # type: ignore[arg-type]


def test_retry_after_seconds_is_none_when_absent_or_invalid() -> None:
    assert _retry_after_seconds(_FakeResponse(429)) is None  # type: ignore[arg-type]
    assert _retry_after_seconds(_FakeResponse(429, headers={"Retry-After": "soon"})) is None  # type: ignore[arg-type]


def test_retry_after_seconds_parses_http_date() -> None:
    future = format_datetime(datetime.now(UTC) + timedelta(seconds=30))
    parsed = _retry_after_seconds(_FakeResponse(429, headers={"Retry-After": future}))  # type: ignore[arg-type]
    assert parsed is not None
    assert 20.0 <= parsed <= 35.0


@pytest.mark.parametrize(
    "value",
    [
        "-5",
        format_datetime(datetime.now(UTC) - timedelta(seconds=60)),  # a past HTTP-date
    ],
)
def test_retry_after_seconds_clamps_past_to_zero(value: str) -> None:
    assert _retry_after_seconds(_FakeResponse(429, headers={"Retry-After": value})) == 0.0  # type: ignore[arg-type]


def test_shed_delay_caps_floors_and_defaults() -> None:
    capped = _shed_delay(_FakeResponse(429, headers={"Retry-After": "999"}))  # type: ignore[arg-type]
    assert capped == pytest.approx(BACKOFF_MAX_SECONDS)
    floored = _shed_delay(_FakeResponse(429, headers={"Retry-After": "0"}))  # type: ignore[arg-type]
    assert floored == pytest.approx(BACKOFF_BASE_SECONDS)  # a zero Retry-After cannot tight-loop
    assert _shed_delay(_FakeResponse(429)) == pytest.approx(BACKOFF_BASE_SECONDS)  # type: ignore[arg-type]


def test_fetch_prediction_retries_on_connection_error() -> None:
    session = _FakeSession([requests.ConnectionError("boom"), _FakeResponse(200, _PREDICT_BODY)])
    result = _fetch_prediction(_cfg(), session, _event(), ShutdownFlag())  # type: ignore[arg-type]
    assert result.outcome is FetchOutcome.OK
    assert session.calls == 2


def test_fetch_prediction_rejects_on_client_error() -> None:
    session = _FakeSession([_FakeResponse(422)])
    result = _fetch_prediction(_cfg(), session, _event(), ShutdownFlag())  # type: ignore[arg-type]
    assert result.outcome is FetchOutcome.REJECTED
    assert session.calls == 1


def test_fetch_prediction_unavailable_after_max_retries() -> None:
    session = _FakeSession([_FakeResponse(503)] * MAX_PREDICT_RETRIES)
    result = _fetch_prediction(_cfg(), session, _event(), ShutdownFlag())  # type: ignore[arg-type]
    assert result.outcome is FetchOutcome.UNAVAILABLE
    assert session.calls == MAX_PREDICT_RETRIES


def test_fetch_prediction_aborts_when_shutdown_requested() -> None:
    shutdown = ShutdownFlag()
    shutdown.requested = True
    session = _FakeSession([_FakeResponse(200, _PREDICT_BODY)])
    result = _fetch_prediction(_cfg(), session, _event(), shutdown)  # type: ignore[arg-type]
    assert result.outcome is FetchOutcome.ABORTED
    assert session.calls == 0


def test_fetch_prediction_sends_bearer_header_when_key_set() -> None:
    session = _FakeSession([_FakeResponse(200, _PREDICT_BODY)])
    cfg = _cfg(predict_api_key="secret")
    _fetch_prediction(cfg, session, _event(), ShutdownFlag())  # type: ignore[arg-type]
    assert session.headers == {"Authorization": "Bearer secret"}


def test_fetch_prediction_omits_auth_header_when_no_key() -> None:
    session = _FakeSession([_FakeResponse(200, _PREDICT_BODY)])
    _fetch_prediction(_cfg(), session, _event(), ShutdownFlag())  # type: ignore[arg-type]
    assert session.headers == {}


@pytest.mark.parametrize("status", [401, 403])
def test_fetch_prediction_holds_without_dlq_on_auth_failure(status: int) -> None:
    # A wrong key must not dead-letter valid transactions; it holds for a fixed key to replay.
    session = _FakeSession([_FakeResponse(status)])
    result = _fetch_prediction(_cfg(), session, _event(), ShutdownFlag())  # type: ignore[arg-type]
    assert result.outcome is FetchOutcome.UNAVAILABLE
    assert session.calls == 1


def _handle(
    message: Any,
    session: Any,
    producer: Any,
    consumer: Any,
    breaker: _CircuitBreaker,
) -> bool:
    return _handle_message(message, _cfg(), session, producer, consumer, ShutdownFlag(), breaker)


def test_handle_message_publishes_prediction_and_commits_on_success() -> None:
    producer, consumer = _FakeProducer(), _FakeConsumer()
    session = _FakeSession([_FakeResponse(200, _PREDICT_BODY)])
    committed = _handle(_FakeMessage(_payload()), session, producer, consumer, _breaker())
    assert committed is True
    assert consumer.commits == 1
    assert producer.produced[0][0] == "predictions"
    assert consumer.seeks == []


def test_handle_message_routes_poison_to_dlq_and_commits() -> None:
    producer, consumer = _FakeProducer(), _FakeConsumer()
    message = _FakeMessage(b"not-json")
    committed = _handle(message, _FakeSession([]), producer, consumer, _breaker())
    assert committed is True
    assert consumer.commits == 1
    topic, _value, headers = producer.produced[0]
    assert topic == "transactions-dlq"
    assert ("reason", b"deserialize_error") in headers


def test_handle_message_routes_rejected_to_dlq_and_commits() -> None:
    producer, consumer = _FakeProducer(), _FakeConsumer()
    session = _FakeSession([_FakeResponse(422)])
    committed = _handle(_FakeMessage(_payload()), session, producer, consumer, _breaker())
    assert committed is True
    assert consumer.commits == 1
    assert producer.produced[0][0] == "transactions-dlq"
    assert ("reason", b"rejected") in producer.produced[0][2]


def test_handle_message_counts_throttle_toward_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fraud.ingestion.consumer._interruptible_sleep", lambda _seconds, _shutdown: None
    )
    producer, consumer = _FakeProducer(), _FakeConsumer()
    session = _FakeSession([_FakeResponse(429)])
    breaker = _breaker()
    committed = _handle(_FakeMessage(_payload()), session, producer, consumer, breaker)
    assert committed is False  # held, not committed
    assert producer.produced == []  # not dead-lettered
    assert breaker.failures == 1  # a single 429 counts toward the breaker
    assert session.calls == 1


def test_handle_message_holds_offset_when_serving_unavailable() -> None:
    producer, consumer = _FakeProducer(), _FakeConsumer()
    session = _FakeSession([_FakeResponse(503)] * MAX_PREDICT_RETRIES)
    breaker = _breaker()
    committed = _handle(_FakeMessage(_payload()), session, producer, consumer, breaker)
    assert committed is False
    assert consumer.commits == 0
    assert producer.produced == []
    assert breaker.failures == 1


def test_handle_message_holds_offset_on_malformed_success_body() -> None:
    # A 2xx that breaks the contract holds the message instead of dropping or dead-lettering it.
    producer, consumer = _FakeProducer(), _FakeConsumer()
    session = _FakeSession([_FakeResponse(200, {"fraud_score": 0.1})])
    breaker = _breaker()
    committed = _handle(_FakeMessage(_payload()), session, producer, consumer, breaker)
    assert committed is False
    assert consumer.commits == 0
    assert producer.produced == []
    assert breaker.failures == 1


def test_circuit_breaker_opens_after_threshold_and_resets_on_success() -> None:
    breaker = _CircuitBreaker(threshold=2, cooldown_seconds=1.0)
    breaker.record_failure()
    one_failure = breaker.is_open
    breaker.record_failure()
    two_failures = breaker.is_open
    breaker.record_success()
    after_reset = breaker.is_open
    assert (one_failure, two_failures, after_reset) == (False, True, False)


def test_seek_back_rewinds_to_message_offset() -> None:
    consumer = _FakeConsumer()
    message = _FakeMessage(b"x", topic="transactions", partition=2, offset=42)
    _seek_back(consumer, message)  # type: ignore[arg-type]
    assert len(consumer.seeks) == 1
    partition = consumer.seeks[0]
    assert (partition.topic, partition.partition, partition.offset) == ("transactions", 2, 42)


def test_handle_message_holds_offset_when_publish_unconfirmed() -> None:
    # Serving answered; only the Kafka publish failed, so hold the offset but leave the breaker.
    producer, consumer = _FakeProducer(deliver=False), _FakeConsumer()
    session = _FakeSession([_FakeResponse(200, _PREDICT_BODY)])
    breaker = _breaker()
    committed = _handle(_FakeMessage(_payload()), session, producer, consumer, breaker)
    assert committed is False
    assert consumer.commits == 0
    assert breaker.failures == 0


def test_dlq_write_does_not_reset_serving_breaker() -> None:
    # Dead-lettering a poison message during a serving outage must leave the breaker armed.
    producer, consumer = _FakeProducer(), _FakeConsumer()
    breaker = _breaker()
    breaker.record_failure()
    breaker.record_failure()
    _handle(_FakeMessage(b"not-json"), _FakeSession([]), producer, consumer, breaker)
    assert breaker.failures == 2


def test_handle_message_holds_offset_on_producer_buffer_error() -> None:
    # A full librdkafka queue must hold the offset, not crash the loop or arm the serving breaker.
    class _BufferFullProducer(_FakeProducer):
        def produce(self, *args: Any, **kwargs: Any) -> None:
            raise BufferError("queue full")

    consumer = _FakeConsumer()
    session = _FakeSession([_FakeResponse(200, _PREDICT_BODY)])
    breaker = _breaker()
    committed = _handle(_FakeMessage(_payload()), session, _BufferFullProducer(), consumer, breaker)
    assert committed is False
    assert consumer.commits == 0
    assert breaker.failures == 0


def test_fetch_prediction_rejects_non_2xx_status() -> None:
    # Only a 2xx carries a prediction; a 3xx from a misrouted endpoint is a permanent reject.
    session = _FakeSession([_FakeResponse(301)])
    result = _fetch_prediction(_cfg(), session, _event(), ShutdownFlag())  # type: ignore[arg-type]
    assert result.outcome is FetchOutcome.REJECTED
    assert session.calls == 1


def test_handle_message_routes_empty_payload_to_dlq() -> None:
    producer, consumer = _FakeProducer(), _FakeConsumer()
    committed = _handle(_FakeMessage(None), _FakeSession([]), producer, consumer, _breaker())
    assert committed is True
    assert consumer.commits == 1
    topic, _value, headers = producer.produced[0]
    assert topic == "transactions-dlq"
    assert ("reason", b"empty_payload") in headers


def test_interruptible_sleep_returns_when_shutdown_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        "fraud.ingestion.consumer.time.sleep", lambda seconds: sleeps.append(seconds)
    )
    shutdown = ShutdownFlag()
    shutdown.requested = True
    _interruptible_sleep(100.0, shutdown)
    assert sleeps == []  # the flag short-circuits the loop before any sleep


def test_run_consumer_holds_offset_and_paces_when_serving_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shutdown = ShutdownFlag()
    consumer = _LoopConsumer(_FakeMessage(_payload()), deliver_count=4, shutdown=shutdown)
    producer = _FakeProducer()
    session = _FakeSession([_FakeResponse(503)] * (MAX_PREDICT_RETRIES * 4))
    cooldowns: list[float] = []
    monkeypatch.setattr("fraud.ingestion.consumer._build_consumer", lambda cfg: consumer)
    monkeypatch.setattr("fraud.ingestion.consumer.Producer", lambda cfg: producer)
    monkeypatch.setattr("fraud.ingestion.consumer.requests.Session", lambda: session)
    monkeypatch.setattr(
        "fraud.ingestion.consumer._interruptible_sleep",
        lambda seconds, _shutdown: cooldowns.append(seconds),
    )
    heartbeat = tmp_path / "consumer-alive"

    run_consumer(_cfg(), shutdown, heartbeat_path=heartbeat)

    assert consumer.commits == 0
    assert consumer.seeks
    assert cooldowns
    assert heartbeat.exists()  # the poll loop refreshed the liveness heartbeat
