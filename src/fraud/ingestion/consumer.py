"""Consume transactions, score them through the serving API, and publish predictions.

At-least-once with bounded retries, a poison-message DLQ, and a circuit breaker around serving.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import requests
from confluent_kafka import Consumer, KafkaException, Message, Producer, TopicPartition
from pydantic import ValidationError

from fraud.common.logging import configure_logging, get_logger
from fraud.common.shutdown import ShutdownFlag, install_shutdown_handler
from fraud.config import get_settings
from fraud.streaming.events import (
    PredictionEvent,
    TransactionEvent,
    deserialize_transaction,
    serialize,
)
from fraud.streaming.transport import StreamConfig, durable_producer_config

log = get_logger(__name__)

POLL_TIMEOUT_SECONDS = 1.0
PREDICT_TIMEOUT_SECONDS = 5.0
MAX_PREDICT_RETRIES = 5
BACKOFF_BASE_SECONDS = 0.5
BACKOFF_MAX_SECONDS = 8.0
BACKOFF_JITTER_SECONDS = 0.3
CLIENT_ERROR_START = 400
SERVER_ERROR_START = 500
TOO_MANY_REQUESTS = 429
UNAUTHORIZED = 401
FORBIDDEN = 403
CIRCUIT_BREAKER_THRESHOLD = 3
CIRCUIT_COOLDOWN_SECONDS = 10.0
SHUTDOWN_POLL_SECONDS = 0.5
FLUSH_TIMEOUT_SECONDS = 10.0
# Touched each poll cycle; a k8s exec probe restarts the pod if it goes stale (wedged poll).
HEARTBEAT_PATH = Path("/tmp/consumer-alive")


class FetchOutcome(StrEnum):
    OK = "ok"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class PredictionResult:
    outcome: FetchOutcome
    prediction: PredictionEvent | None = None


@dataclass(slots=True)
class _CircuitBreaker:
    threshold: int
    cooldown_seconds: float
    failures: int = 0

    def record_success(self) -> None:
        self.failures = 0

    def record_failure(self) -> None:
        self.failures += 1

    @property
    def is_open(self) -> bool:
        return self.failures >= self.threshold


def run_consumer(
    cfg: StreamConfig,
    shutdown: ShutdownFlag | None = None,
    heartbeat_path: Path = HEARTBEAT_PATH,
) -> None:
    """Consume transactions, score each via the serving API, publish predictions."""
    shutdown = shutdown or install_shutdown_handler()
    consumer = _build_consumer(cfg)
    producer = Producer(durable_producer_config(cfg.bootstrap_servers))
    session = requests.Session()
    breaker = _CircuitBreaker(CIRCUIT_BREAKER_THRESHOLD, CIRCUIT_COOLDOWN_SECONDS)
    consumer.subscribe([cfg.transactions_topic])
    log.info("consumer_start", topic=cfg.transactions_topic, predict_url=cfg.predict_url)

    try:
        while not shutdown.requested:
            message = consumer.poll(POLL_TIMEOUT_SECONDS)
            _touch_heartbeat(heartbeat_path)
            if message is None:
                continue
            if message.error():
                log.warning("consume_error", error=str(message.error()))
                continue
            committed = _handle_message(
                message, cfg, session, producer, consumer, shutdown, breaker
            )
            if not committed and not shutdown.requested:
                _seek_back(consumer, message)
                if breaker.is_open:
                    log.warning("downstream_circuit_open", failures=breaker.failures)
                    _interruptible_sleep(breaker.cooldown_seconds, shutdown)
    finally:
        consumer.close()
        producer.flush(FLUSH_TIMEOUT_SECONDS)
        session.close()
        log.info("consumer_stopped")


def _build_consumer(cfg: StreamConfig) -> Consumer:
    return Consumer(
        {
            "bootstrap.servers": cfg.bootstrap_servers,
            "group.id": cfg.consumer_group,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
        }
    )


def _handle_message(
    message: Message,
    cfg: StreamConfig,
    session: requests.Session,
    producer: Producer,
    consumer: Consumer,
    shutdown: ShutdownFlag,
    breaker: _CircuitBreaker,
) -> bool:
    """Score one message; True if committed, False to redeliver when a downstream is down."""
    payload = message.value()
    if payload is None:
        log.warning("empty_payload_routed_to_dlq")
        return _commit_to_dlq(message, cfg, producer, consumer, breaker, reason="empty_payload")
    try:
        event = deserialize_transaction(payload)
    except ValidationError as exc:
        log.warning("poison_message_routed_to_dlq", error=str(exc))
        return _commit_to_dlq(message, cfg, producer, consumer, breaker, reason="deserialize_error")

    result = _fetch_prediction(cfg, session, event, shutdown, max_attempts=_probe_or_full(breaker))
    if result.outcome is FetchOutcome.OK and result.prediction is not None:
        return _commit_prediction(message, cfg, producer, consumer, breaker, result.prediction)
    if result.outcome is FetchOutcome.REJECTED:
        log.warning("prediction_rejected_to_dlq", transaction_id=event.transaction_id)
        return _commit_to_dlq(message, cfg, producer, consumer, breaker, reason="rejected")
    if result.outcome is FetchOutcome.UNAVAILABLE:
        breaker.record_failure()
    return False


def _commit_prediction(
    message: Message,
    cfg: StreamConfig,
    producer: Producer,
    consumer: Consumer,
    breaker: _CircuitBreaker,
    prediction: PredictionEvent,
) -> bool:
    if not _publish_prediction(producer, cfg, prediction):
        breaker.record_failure()
        return False
    _note_recovery(breaker)
    consumer.commit(message=message, asynchronous=False)
    log.info(
        "prediction_consumed",
        transaction_id=prediction.transaction_id,
        decision=prediction.decision,
        fraud_score=prediction.fraud_score,
    )
    return True


def _commit_to_dlq(
    message: Message,
    cfg: StreamConfig,
    producer: Producer,
    consumer: Consumer,
    breaker: _CircuitBreaker,
    *,
    reason: str,
) -> bool:
    if not _to_dlq(producer, cfg, message, reason=reason):
        breaker.record_failure()
        return False
    _note_recovery(breaker)
    consumer.commit(message=message, asynchronous=False)
    return True


def _probe_or_full(breaker: _CircuitBreaker) -> int:
    # While open, one probe per cooldown checks for recovery instead of re-running the full budget.
    return 1 if breaker.is_open else MAX_PREDICT_RETRIES


def _note_recovery(breaker: _CircuitBreaker) -> None:
    if breaker.is_open:
        log.info("downstream_recovered", failures=breaker.failures)
    breaker.record_success()


def _fetch_prediction(
    cfg: StreamConfig,
    session: requests.Session,
    event: TransactionEvent,
    shutdown: ShutdownFlag,
    *,
    max_attempts: int = MAX_PREDICT_RETRIES,
) -> PredictionResult:
    """Bounded retries: 5xx is retried, 429 honors Retry-After and yields to the breaker, 401/403
    holds, and any other 4xx is a permanent reject."""
    body = predict_request_body(event)
    headers = _auth_headers(cfg)
    for attempt in range(1, max_attempts + 1):
        if shutdown.requested:
            return PredictionResult(FetchOutcome.ABORTED)
        try:
            response = session.post(
                cfg.predict_url, json=body, headers=headers, timeout=PREDICT_TIMEOUT_SECONDS
            )
        except requests.RequestException as exc:
            log.warning("predict_call_failed", attempt=attempt, error=str(exc))
            _backoff(attempt)
            continue
        if response.status_code < CLIENT_ERROR_START:
            return PredictionResult(
                FetchOutcome.OK, prediction_from_response(event, response.json())
            )
        if response.status_code == TOO_MANY_REQUESTS:
            # Shedding: honor Retry-After and let the breaker open now instead of burning the
            # whole retry budget per message, which would only pile more load on the server.
            log.warning("predict_throttled", attempt=attempt)
            _interruptible_sleep(_shed_delay(response), shutdown)
            return PredictionResult(FetchOutcome.UNAVAILABLE)
        if _is_retryable_status(response.status_code):
            log.warning("predict_unavailable", status=response.status_code, attempt=attempt)
            _backoff(attempt)
            continue
        if response.status_code in (UNAUTHORIZED, FORBIDDEN):
            # An auth failure is our own misconfiguration, not a poison payload; hold the message
            # so a fixed key replays it instead of dead-lettering a stream of valid transactions.
            log.error("predict_auth_rejected", status=response.status_code)
            return PredictionResult(FetchOutcome.UNAVAILABLE)
        log.warning(
            "predict_rejected", status=response.status_code, transaction_id=event.transaction_id
        )
        return PredictionResult(FetchOutcome.REJECTED)

    log.error("predict_retries_exhausted", transaction_id=event.transaction_id)
    return PredictionResult(FetchOutcome.UNAVAILABLE)


def _is_retryable_status(status_code: int) -> bool:
    return status_code >= SERVER_ERROR_START


def _shed_delay(response: requests.Response) -> float:
    retry_after = _retry_after_seconds(response)
    if retry_after is None:
        return BACKOFF_BASE_SECONDS
    # Floor a zero or past Retry-After to the base backoff so a server cannot drive a tight retry.
    return min(max(retry_after, BACKOFF_BASE_SECONDS), BACKOFF_MAX_SECONDS)


def _retry_after_seconds(response: requests.Response) -> float | None:
    """Parse a Retry-After header in either delta-seconds or HTTP-date form; None if absent."""
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def _touch_heartbeat(path: Path) -> None:
    try:
        path.touch()
    except OSError as exc:
        log.warning("heartbeat_touch_failed", error=str(exc))


def _auth_headers(cfg: StreamConfig) -> dict[str, str]:
    if cfg.predict_api_key is None:
        return {}
    return {"Authorization": f"Bearer {cfg.predict_api_key}"}


def predict_request_body(event: TransactionEvent) -> dict[str, dict[str, Any]]:
    return {
        "request": {
            "card_id": event.card_id,
            "amount": event.amount,
            "transaction_id": event.transaction_id,
            "raw": event.raw.model_dump(),
        }
    }


def prediction_from_response(event: TransactionEvent, body: dict[str, Any]) -> PredictionEvent:
    return PredictionEvent(
        transaction_id=event.transaction_id,
        card_id=event.card_id,
        fraud_score=float(body["fraud_score"]),
        decision=bool(body["decision"]),
        threshold=float(body["threshold"]),
        model_version=int(body["model_version"]),
    )


def _publish_prediction(producer: Producer, cfg: StreamConfig, prediction: PredictionEvent) -> bool:
    return _produce_confirmed(
        producer,
        cfg.predictions_topic,
        key=prediction.card_id.encode("utf-8"),
        value=serialize(prediction),
    )


def _to_dlq(producer: Producer, cfg: StreamConfig, message: Message, *, reason: str) -> bool:
    headers: list[tuple[str, str | bytes | None]] = [("reason", reason.encode("utf-8"))]
    return _produce_confirmed(
        producer,
        cfg.dlq_topic,
        key=message.key(),
        value=message.value(),
        headers=headers,
    )


def _produce_confirmed(
    producer: Producer,
    topic: str,
    *,
    key: bytes | None,
    value: bytes | None,
    headers: list[tuple[str, str | bytes | None]] | None = None,
) -> bool:
    # flush() reports a permanent failure only via the callback, so confirm both: queue drained
    # within the timeout AND no delivery error. An unconfirmed write must not advance the offset.
    delivered = True

    def _on_delivery(error: object, _message: object) -> None:
        nonlocal delivered
        if error is not None:
            delivered = False
            log.warning("delivery_failed", topic=topic, error=str(error))

    producer.produce(topic, key=key, value=value, headers=headers, on_delivery=_on_delivery)
    pending = producer.flush(FLUSH_TIMEOUT_SECONDS)
    return pending == 0 and delivered


def _seek_back(consumer: Consumer, message: Message) -> None:
    # A transaction we could not score is re-delivered once a downstream recovers, never dropped.
    topic, partition, offset = message.topic(), message.partition(), message.offset()
    if topic is None or partition is None or offset is None:
        return
    try:
        consumer.seek(TopicPartition(topic, partition, offset))
    except KafkaException as exc:
        # A rebalance can revoke the partition between poll and seek; its new owner redelivers.
        log.warning("seek_back_failed", error=str(exc))


def _backoff(attempt: int) -> None:
    delay = min(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), BACKOFF_MAX_SECONDS)
    time.sleep(delay + random.uniform(0.0, BACKOFF_JITTER_SECONDS))


def _interruptible_sleep(seconds: float, shutdown: ShutdownFlag) -> None:
    deadline = time.monotonic() + seconds
    while not shutdown.requested:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return
        time.sleep(min(SHUTDOWN_POLL_SECONDS, remaining))


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    cfg = StreamConfig.from_settings()
    run_consumer(cfg)


if __name__ == "__main__":
    main()
