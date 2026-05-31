from __future__ import annotations

import signal
import time
from types import FrameType
from typing import Any

import requests
from confluent_kafka import Consumer, Message, Producer
from pydantic import ValidationError

from fraud.common.logging import configure_logging, get_logger
from fraud.config import get_settings
from fraud.ingestion.stream import (
    PredictionEvent,
    StreamConfig,
    TransactionEvent,
    deserialize_transaction,
    serialize,
)

log = get_logger(__name__)

POLL_TIMEOUT_SECONDS = 1.0
PREDICT_TIMEOUT_SECONDS = 5.0
MAX_PREDICT_RETRIES = 5
BACKOFF_BASE_SECONDS = 0.5
BACKOFF_MAX_SECONDS = 8.0
CLIENT_ERROR_START = 400
SERVER_ERROR_START = 500


class _ShutdownFlag:
    def __init__(self) -> None:
        self.requested = False

    def request(self, _signum: int, _frame: FrameType | None) -> None:
        self.requested = True


def run_consumer(cfg: StreamConfig, shutdown: _ShutdownFlag | None = None) -> None:
    """Consume transactions, score each via the serving API, publish predictions."""
    shutdown = shutdown or _install_shutdown_handler()
    consumer = _build_consumer(cfg)
    producer = Producer({"bootstrap.servers": cfg.bootstrap_servers})
    session = requests.Session()
    consumer.subscribe([cfg.transactions_topic])
    log.info("consumer_start", topic=cfg.transactions_topic, predict_url=cfg.predict_url)

    try:
        while not shutdown.requested:
            message = consumer.poll(POLL_TIMEOUT_SECONDS)
            if message is None:
                continue
            if message.error():
                log.warning("consume_error", error=str(message.error()))
                continue
            _handle_message(message, cfg, session, producer, consumer, shutdown)
    finally:
        consumer.close()
        producer.flush()
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
    shutdown: _ShutdownFlag,
) -> None:
    payload = message.value()
    if payload is None:
        consumer.commit(message=message, asynchronous=False)
        return
    try:
        event = deserialize_transaction(payload)
    except ValidationError as exc:
        log.warning("poison_message_skipped", error=str(exc))
        consumer.commit(message=message, asynchronous=False)
        return

    prediction = _fetch_prediction(cfg, session, event, shutdown)
    if prediction is None:
        # Commit to skip a rejected/unreachable message; on shutdown leave it for redelivery.
        if not shutdown.requested:
            consumer.commit(message=message, asynchronous=False)
        return

    producer.produce(
        cfg.predictions_topic,
        key=prediction.card_id.encode("utf-8"),
        value=serialize(prediction),
    )
    producer.flush()
    consumer.commit(message=message, asynchronous=False)
    log.info(
        "prediction_consumed",
        transaction_id=prediction.transaction_id,
        decision=prediction.decision,
        fraud_score=prediction.fraud_score,
    )


def _fetch_prediction(
    cfg: StreamConfig,
    session: requests.Session,
    event: TransactionEvent,
    shutdown: _ShutdownFlag,
) -> PredictionEvent | None:
    body = predict_request_body(event)
    for attempt in range(1, MAX_PREDICT_RETRIES + 1):
        if shutdown.requested:
            return None
        try:
            response = session.post(cfg.predict_url, json=body, timeout=PREDICT_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            log.warning("predict_call_failed", attempt=attempt, error=str(exc))
            _backoff(attempt)
            continue
        if response.status_code < CLIENT_ERROR_START:
            return prediction_from_response(event, response.json())
        if response.status_code < SERVER_ERROR_START:
            log.warning(
                "predict_rejected",
                status=response.status_code,
                transaction_id=event.transaction_id,
            )
            return None
        log.warning("predict_unavailable", status=response.status_code, attempt=attempt)
        _backoff(attempt)

    log.error("predict_retries_exhausted", transaction_id=event.transaction_id)
    return None


def predict_request_body(event: TransactionEvent) -> dict[str, dict[str, Any]]:
    return {
        "request": {
            "card_id": event.card_id,
            "amount": event.amount,
            "transaction_id": event.transaction_id,
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


def _backoff(attempt: int) -> None:
    delay = min(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), BACKOFF_MAX_SECONDS)
    time.sleep(delay)


def _install_shutdown_handler() -> _ShutdownFlag:
    shutdown = _ShutdownFlag()
    signal.signal(signal.SIGINT, shutdown.request)
    signal.signal(signal.SIGTERM, shutdown.request)
    return shutdown


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    cfg = StreamConfig.from_settings()
    run_consumer(cfg)


if __name__ == "__main__":
    main()
