from __future__ import annotations

import signal
import time
from collections.abc import Iterator
from pathlib import Path
from types import FrameType

import pandas as pd
from confluent_kafka import Producer

from fraud.common.logging import configure_logging, get_logger
from fraud.config import get_settings
from fraud.ingestion.stream import StreamConfig, TransactionEvent, seconds_per_message, serialize
from fraud.paths import PROCESSED_DIR
from fraud.transforms import feature_logic as fl

log = get_logger(__name__)

_SOURCE_COLUMNS = ["TransactionID", "TransactionDT", "TransactionAmt", *fl.IDENTITY_COLUMNS]


class _ShutdownFlag:
    def __init__(self) -> None:
        self.requested = False

    def request(self, _signum: int, _frame: FrameType | None) -> None:
        self.requested = True


def run_producer(cfg: StreamConfig, source_path: Path) -> int:
    """Replay transactions from the parquet to the topic at the configured rate."""
    frame = _load_transactions(source_path)
    producer = Producer({"bootstrap.servers": cfg.bootstrap_servers})
    delay = seconds_per_message(cfg.replay_rate_per_second)
    shutdown = _install_shutdown_handler()

    published = 0
    try:
        for event in _iter_events(frame):
            if shutdown.requested:
                break
            _publish(producer, cfg.transactions_topic, event)
            producer.poll(0)
            published += 1
            time.sleep(delay)
    finally:
        producer.flush()
    log.info("replay_complete", published=published, topic=cfg.transactions_topic)
    return published


def _load_transactions(source_path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(source_path, columns=_SOURCE_COLUMNS)
    keyed = fl.add_keys_and_timestamp(raw)
    keyed["event_timestamp"] = keyed["event_timestamp"].map(lambda ts: ts.isoformat())
    return keyed


def _iter_events(frame: pd.DataFrame) -> Iterator[TransactionEvent]:
    columns = ["TransactionID", "card_id", "TransactionAmt", "event_timestamp"]
    for record in frame[columns].to_dict("records"):
        yield TransactionEvent(
            transaction_id=str(record["TransactionID"]),
            card_id=str(record["card_id"]),
            amount=float(record["TransactionAmt"]),
            event_timestamp=str(record["event_timestamp"]),
        )


def _publish(producer: Producer, topic: str, event: TransactionEvent) -> None:
    producer.produce(
        topic,
        key=event.card_id.encode("utf-8"),
        value=serialize(event),
        on_delivery=_on_delivery,
    )


def _on_delivery(error: object, _message: object) -> None:
    if error is not None:
        log.warning("delivery_failed", error=str(error))


def _install_shutdown_handler() -> _ShutdownFlag:
    shutdown = _ShutdownFlag()
    signal.signal(signal.SIGINT, shutdown.request)
    signal.signal(signal.SIGTERM, shutdown.request)
    return shutdown


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    cfg = StreamConfig.from_settings()
    log.info("producer_start", rate=cfg.replay_rate_per_second, topic=cfg.transactions_topic)
    run_producer(cfg, PROCESSED_DIR / "test.parquet")


if __name__ == "__main__":
    main()
