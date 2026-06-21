"""Replay ground-truth labels onto the labels topic after a delay, modeling chargeback lag."""

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
from fraud.ingestion.stream import (
    LABELS_TOPIC,
    LabelEvent,
    StreamConfig,
    seconds_per_message,
    serialize,
)
from fraud.paths import PROCESSED_DIR
from fraud.training.features import LABEL_COLUMN

log = get_logger(__name__)

_SOURCE_COLUMNS = ["TransactionID", LABEL_COLUMN]
_LEAD_STEP_SECONDS = 0.25


class _ShutdownFlag:
    def __init__(self) -> None:
        self.requested = False

    def request(self, _signum: int, _frame: FrameType | None) -> None:
        self.requested = True


def run_label_simulator(
    cfg: StreamConfig,
    source_path: Path,
    *,
    lead_seconds: float,
    producer: Producer | None = None,
    shutdown: _ShutdownFlag | None = None,
) -> int:
    """Emit each transaction's true label to the labels topic, trailing its prediction."""
    frame = _load_labels(source_path)
    producer = producer or Producer({"bootstrap.servers": cfg.bootstrap_servers})
    shutdown = shutdown or _install_shutdown_handler()
    delay = seconds_per_message(cfg.replay_rate_per_second)

    _sleep_interruptible(lead_seconds, shutdown)
    published = 0
    try:
        for event in _iter_labels(frame):
            if shutdown.requested:
                break
            _publish(producer, LABELS_TOPIC, event)
            producer.poll(0)
            published += 1
            time.sleep(delay)
    finally:
        producer.flush()
    log.info("labels_complete", published=published, topic=LABELS_TOPIC)
    return published


def _load_labels(source_path: Path) -> pd.DataFrame:
    return pd.read_parquet(source_path, columns=_SOURCE_COLUMNS)


def _iter_labels(frame: pd.DataFrame) -> Iterator[LabelEvent]:
    for record in frame.to_dict("records"):
        yield LabelEvent(
            transaction_id=str(record["TransactionID"]),
            is_fraud=int(record[LABEL_COLUMN]),
        )


def _publish(producer: Producer, topic: str, event: LabelEvent) -> None:
    producer.produce(
        topic,
        key=event.transaction_id.encode("utf-8"),
        value=serialize(event),
    )


def _sleep_interruptible(seconds: float, shutdown: _ShutdownFlag) -> None:
    deadline = time.monotonic() + seconds
    while not shutdown.requested and time.monotonic() < deadline:
        time.sleep(min(_LEAD_STEP_SECONDS, deadline - time.monotonic()))


def _install_shutdown_handler() -> _ShutdownFlag:
    shutdown = _ShutdownFlag()
    signal.signal(signal.SIGINT, shutdown.request)
    signal.signal(signal.SIGTERM, shutdown.request)
    return shutdown


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    cfg = StreamConfig.from_settings()
    log.info(
        "label_simulator_start",
        rate=cfg.replay_rate_per_second,
        lead_seconds=settings.stream_label_delay_seconds,
        topic=LABELS_TOPIC,
    )
    # Trail labels for the holdout the producer replays.
    run_label_simulator(
        cfg, PROCESSED_DIR / "holdout.parquet", lead_seconds=settings.stream_label_delay_seconds
    )


if __name__ == "__main__":
    main()
