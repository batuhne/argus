"""Replay ground-truth labels onto the labels topic, each after its time-warped chargeback lag."""

from __future__ import annotations

import signal
import time
from pathlib import Path
from types import FrameType

import numpy as np
import pandas as pd
from confluent_kafka import Producer

from fraud.common.logging import configure_logging, get_logger
from fraud.config import get_settings
from fraud.ingestion.stream import (
    LABELS_TOPIC,
    SECONDS_PER_DAY,
    LabelEvent,
    StreamConfig,
    replay_step_delays,
    serialize,
)
from fraud.params import StreamParams, load_params
from fraud.paths import PROCESSED_DIR
from fraud.training.features import LABEL_COLUMN

log = get_logger(__name__)

_SOURCE_COLUMNS = ["TransactionID", LABEL_COLUMN, "TransactionDT"]
_SLEEP_GRANULARITY_SECONDS = 0.25


class _ShutdownFlag:
    def __init__(self) -> None:
        self.requested = False

    def request(self, _signum: int, _frame: FrameType | None) -> None:
        self.requested = True


def run_label_simulator(
    cfg: StreamConfig,
    source_path: Path,
    *,
    stream_params: StreamParams,
    seed: int,
    producer: Producer | None = None,
    shutdown: _ShutdownFlag | None = None,
) -> int:
    """Emit each transaction's true label after its time-warped chargeback lag."""
    frame = _load_labels(source_path)
    producer = producer or Producer({"bootstrap.servers": cfg.bootstrap_servers})
    shutdown = shutdown or _install_shutdown_handler()
    schedule = _label_schedule(frame, stream_params, seed)

    start = time.monotonic()
    published = 0
    try:
        for emit_offset, event in schedule:
            # The sleep returns early on shutdown, so a single post-wait check stops promptly.
            _sleep_interruptible(start + emit_offset - time.monotonic(), shutdown)
            if shutdown.requested:
                break
            _publish(producer, LABELS_TOPIC, event)
            producer.poll(0)
            published += 1
    finally:
        producer.flush()
    log.info("labels_complete", published=published, topic=LABELS_TOPIC)
    return published


def _load_labels(source_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(source_path, columns=_SOURCE_COLUMNS)
    return frame.sort_values("TransactionDT").reset_index(drop=True)


def _label_schedule(
    frame: pd.DataFrame, stream_params: StreamParams, seed: int
) -> list[tuple[float, LabelEvent]]:
    """Each label's emit offset from start: producer tempo plus a jittered warped lag, sorted."""
    emit_tempo = np.cumsum(
        replay_step_delays(
            frame["TransactionDT"],
            time_warp_factor=stream_params.time_warp_factor,
            max_step_seconds=stream_params.max_message_delay_seconds,
        )
    )
    base_lag = (
        stream_params.base_chargeback_lag_days * SECONDS_PER_DAY / stream_params.time_warp_factor
    )
    jitter = np.random.default_rng(seed).uniform(
        1.0 - stream_params.label_lag_jitter, 1.0 + stream_params.label_lag_jitter, size=len(frame)
    )
    emit_offset = emit_tempo + base_lag * jitter
    events = [
        LabelEvent(transaction_id=str(tid), is_fraud=int(is_fraud))
        for tid, is_fraud in zip(frame["TransactionID"], frame[LABEL_COLUMN], strict=True)
    ]
    return sorted(zip(emit_offset.tolist(), events, strict=True), key=lambda item: item[0])


def _publish(producer: Producer, topic: str, event: LabelEvent) -> None:
    producer.produce(
        topic,
        key=event.transaction_id.encode("utf-8"),
        value=serialize(event),
    )


def _sleep_interruptible(seconds: float, shutdown: _ShutdownFlag) -> None:
    deadline = time.monotonic() + seconds
    while not shutdown.requested:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return
        time.sleep(min(_SLEEP_GRANULARITY_SECONDS, remaining))


def _install_shutdown_handler() -> _ShutdownFlag:
    shutdown = _ShutdownFlag()
    signal.signal(signal.SIGINT, shutdown.request)
    signal.signal(signal.SIGTERM, shutdown.request)
    return shutdown


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    cfg = StreamConfig.from_settings()
    stream_params = load_params().stream
    log.info(
        "label_simulator_start",
        base_lag_days=stream_params.base_chargeback_lag_days,
        warp=stream_params.time_warp_factor,
        topic=LABELS_TOPIC,
    )
    # Trail labels for the holdout the producer replays.
    run_label_simulator(
        cfg, PROCESSED_DIR / "holdout.parquet", stream_params=stream_params, seed=settings.seed
    )


if __name__ == "__main__":
    main()
