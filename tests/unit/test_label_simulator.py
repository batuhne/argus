from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from fraud.common.shutdown import ShutdownFlag
from fraud.ingestion.label_simulator import (
    _label_schedule,
    run_label_simulator,
)
from fraud.params import StreamParams
from fraud.streaming.events import LABELS_TOPIC, deserialize_label
from fraud.streaming.transport import StreamConfig


def _cfg() -> StreamConfig:
    return StreamConfig(
        bootstrap_servers="localhost:19092",
        transactions_topic="transactions",
        predictions_topic="predictions",
        dlq_topic="transactions-dlq",
        consumer_group="argus-fraud-consumer",
        predict_url="http://localhost:3001/predict",
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [101, 102, 103],
            "isFraud": [0, 1, 0],
            "TransactionDT": [10, 20, 30],
        }
    )


def _instant_params() -> StreamParams:
    # max_step 0 zeroes the tempo and lag 0 means every label is due immediately.
    return StreamParams(
        time_warp_factor=1.0,
        base_chargeback_lag_days=0.0,
        label_lag_jitter=0.0,
        max_message_delay_seconds=0.0,
    )


class _FakeProducer:
    def __init__(self) -> None:
        self.produced: list[tuple[str, bytes, bytes]] = []
        self.flushed = False
        self._callbacks: list[Any] = []

    def produce(self, topic: str, key: bytes, value: bytes, on_delivery: Any = None) -> None:
        self.produced.append((topic, key, value))
        if on_delivery is not None:
            self._callbacks.append(on_delivery)

    def poll(self, _timeout: float) -> int:
        return 0

    def flush(self, *_args: object) -> int:
        self.flushed = True
        for callback in self._callbacks:
            callback(None, None)
        self._callbacks.clear()
        return 0


def test_label_schedule_applies_tempo_and_lag_in_order() -> None:
    frame = pd.DataFrame(
        {"TransactionID": [1, 2, 3], "isFraud": [0, 1, 0], "TransactionDT": [100, 110, 140]}
    )
    params = StreamParams(
        time_warp_factor=10.0,
        base_chargeback_lag_days=0.0,
        label_lag_jitter=0.0,
        max_message_delay_seconds=100.0,
    )

    schedule = _label_schedule(frame, params, seed=0)

    # Gaps 10s and 30s warped by 10 give per-message waits 1s and 3s, cumulative [0, 1, 4].
    assert [offset for offset, _ in schedule] == pytest.approx([0.0, 1.0, 4.0])
    assert [event.transaction_id for _, event in schedule] == ["1", "2", "3"]


def test_label_schedule_adds_base_chargeback_lag() -> None:
    frame = pd.DataFrame({"TransactionID": [1, 2], "isFraud": [0, 1], "TransactionDT": [0, 86400]})
    params = StreamParams(
        time_warp_factor=86400.0,
        base_chargeback_lag_days=2.0,
        label_lag_jitter=0.0,
        max_message_delay_seconds=100.0,
    )

    schedule = _label_schedule(frame, params, seed=0)

    # Tempo cumulative [0, 1]; a 2-day lag warped by 86400 is a constant 2s offset.
    assert [offset for offset, _ in schedule] == pytest.approx([2.0, 3.0])


def test_run_publishes_every_label_keyed_by_transaction(tmp_path: Path) -> None:
    source = tmp_path / "holdout.parquet"
    _frame().to_parquet(source)
    producer = _FakeProducer()

    published = run_label_simulator(
        _cfg(),
        source,
        stream_params=_instant_params(),
        seed=0,
        producer=producer,  # type: ignore[arg-type]
        shutdown=ShutdownFlag(),
    )

    assert published == 3
    assert producer.flushed
    assert {topic for topic, _, _ in producer.produced} == {LABELS_TOPIC}
    _, first_key, first_value = producer.produced[0]
    assert first_key == b"101"
    assert deserialize_label(first_value).is_fraud == 0


def test_failed_delivery_is_not_counted(tmp_path: Path) -> None:
    source = tmp_path / "holdout.parquet"
    _frame().to_parquet(source)

    class _FailingProducer(_FakeProducer):
        def flush(self, *_args: object) -> int:
            self.flushed = True
            for callback in self._callbacks:
                callback("broker down", None)
            self._callbacks.clear()
            return 0

    published = run_label_simulator(
        _cfg(),
        source,
        stream_params=_instant_params(),
        seed=0,
        producer=_FailingProducer(),  # type: ignore[arg-type]
        shutdown=ShutdownFlag(),
    )

    assert published == 0  # a delivery error must not count toward the delivered total


def test_shutdown_before_start_publishes_nothing(tmp_path: Path) -> None:
    source = tmp_path / "holdout.parquet"
    _frame().to_parquet(source)
    producer = _FakeProducer()
    shutdown = ShutdownFlag()
    shutdown.requested = True

    published = run_label_simulator(
        _cfg(),
        source,
        stream_params=_instant_params(),
        seed=0,
        producer=producer,  # type: ignore[arg-type]
        shutdown=shutdown,
    )

    assert published == 0
    assert producer.produced == []
