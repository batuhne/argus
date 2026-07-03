"""Replay the holdout transactions onto the stream, keyed by card to keep per-card order."""

from __future__ import annotations

import math
import time
from collections.abc import Hashable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
from confluent_kafka import Producer

from fraud.common.logging import configure_logging, get_logger
from fraud.common.shutdown import install_shutdown_handler
from fraud.config import get_settings
from fraud.params import StreamParams, load_params
from fraud.paths import PROCESSED_DIR
from fraud.streaming.events import RawAttributes, TransactionEvent, serialize
from fraud.streaming.transport import (
    StreamConfig,
    durable_producer_config,
    replay_step_delays,
)
from fraud.transforms import feature_logic as fl

log = get_logger(__name__)

FLUSH_TIMEOUT_SECONDS = 10.0

_SOURCE_COLUMNS = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    *fl.IDENTITY_COLUMNS,
    *fl.RAW_NUMERIC_PASSTHROUGH,
    *fl.V_SELECTED,
    *fl.CATEGORICAL_COLUMNS,
]


def run_producer(cfg: StreamConfig, source_path: Path, *, stream_params: StreamParams) -> int:
    """Replay transactions to the topic at the real per-transaction tempo (time-warped)."""
    frame = _load_transactions(source_path)
    producer = Producer(durable_producer_config(cfg.bootstrap_servers))
    delays = replay_step_delays(
        frame["TransactionDT"],
        time_warp_factor=stream_params.time_warp_factor,
        max_step_seconds=stream_params.max_message_delay_seconds,
    )
    shutdown = install_shutdown_handler()

    delivered = 0

    def _on_delivery(error: object, _message: object) -> None:
        nonlocal delivered
        if error is not None:
            log.warning("delivery_failed", error=str(error))
            return
        delivered += 1

    try:
        for delay, event in zip(delays, _iter_events(frame), strict=True):
            if shutdown.requested:
                break
            if delay > 0.0:
                time.sleep(float(delay))
            _publish(producer, cfg.transactions_topic, event, _on_delivery)
            producer.poll(0)
    finally:
        pending = producer.flush(FLUSH_TIMEOUT_SECONDS)
        if pending:
            log.warning("producer_flush_timeout", pending=pending)
    log.info("replay_complete", delivered=delivered, topic=cfg.transactions_topic)
    return delivered


def _load_transactions(source_path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(source_path, columns=_SOURCE_COLUMNS)
    # Tempo needs ascending TransactionDT; sort here rather than trust the file's order.
    raw = raw.sort_values("TransactionDT").reset_index(drop=True)
    keyed = fl.add_keys_and_timestamp(raw)
    keyed["event_timestamp"] = keyed["event_timestamp"].map(lambda ts: ts.isoformat())
    return keyed


def _iter_events(frame: pd.DataFrame) -> Iterator[TransactionEvent]:
    columns = [
        "TransactionID",
        "card_id",
        "TransactionAmt",
        "event_timestamp",
        *fl.RAW_NUMERIC_PASSTHROUGH,
        *fl.V_SELECTED,
        *fl.CATEGORICAL_COLUMNS,
    ]
    for record in frame[columns].to_dict("records"):
        yield TransactionEvent(
            transaction_id=str(record["TransactionID"]),
            card_id=str(record["card_id"]),
            amount=float(record["TransactionAmt"]),
            event_timestamp=str(record["event_timestamp"]),
            raw=_raw_attributes(record),
        )


def _raw_attributes(record: dict[Hashable, Any]) -> RawAttributes:
    numeric = {column: _optional_float(record[column]) for column in fl.RAW_NUMERIC_PASSTHROUGH}
    v = {column: _optional_float(record[column]) for column in fl.V_SELECTED}
    categorical = {column: _optional_str(record[column]) for column in fl.CATEGORICAL_COLUMNS}
    return RawAttributes(**numeric, v=v, categorical=categorical)


def _optional_float(value: Any) -> float | None:
    # blank or non-finite has no usable number; send null
    if pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _optional_str(value: Any) -> str | None:
    # A blank categorical stays null; the encoder maps null to its missing-category value.
    return None if pd.isna(value) else str(value)


def _publish(producer: Producer, topic: str, event: TransactionEvent, on_delivery: Any) -> None:
    producer.produce(
        topic,
        key=event.card_id.encode("utf-8"),
        value=serialize(event),
        on_delivery=on_delivery,
    )


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    cfg = StreamConfig.from_settings()
    stream_params = load_params().stream
    log.info("producer_start", warp=stream_params.time_warp_factor, topic=cfg.transactions_topic)
    # Replay the holdout so the demo and monitoring run on unseen data.
    run_producer(cfg, PROCESSED_DIR / "holdout.parquet", stream_params=stream_params)


if __name__ == "__main__":
    main()
