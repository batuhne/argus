from pathlib import Path

import pandas as pd

from fraud.ingestion.label_simulator import (
    _iter_labels,
    _ShutdownFlag,
    run_label_simulator,
)
from fraud.ingestion.stream import LABELS_TOPIC, StreamConfig, deserialize_label


def _cfg() -> StreamConfig:
    return StreamConfig(
        bootstrap_servers="localhost:19092",
        transactions_topic="transactions",
        predictions_topic="predictions",
        consumer_group="argus-fraud-consumer",
        replay_rate_per_second=1000.0,
        predict_url="http://localhost:3001/predict",
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"TransactionID": [101, 102, 103], "isFraud": [0, 1, 0]})


class _FakeProducer:
    def __init__(self) -> None:
        self.produced: list[tuple[str, bytes, bytes]] = []
        self.flushed = False

    def produce(self, topic: str, key: bytes, value: bytes) -> None:
        self.produced.append((topic, key, value))

    def poll(self, _timeout: float) -> int:
        return 0

    def flush(self, *_args: object) -> int:
        self.flushed = True
        return 0


def test_iter_labels_maps_transaction_id_and_label() -> None:
    events = list(_iter_labels(_frame()))
    assert [e.transaction_id for e in events] == ["101", "102", "103"]
    assert [e.is_fraud for e in events] == [0, 1, 0]


def test_run_publishes_every_label_keyed_by_transaction(tmp_path: Path) -> None:
    source = tmp_path / "test.parquet"
    _frame().to_parquet(source)
    producer = _FakeProducer()

    published = run_label_simulator(
        _cfg(),
        source,
        lead_seconds=0.0,
        producer=producer,  # type: ignore[arg-type]
        shutdown=_ShutdownFlag(),
    )

    assert published == 3
    assert producer.flushed
    topics = {topic for topic, _, _ in producer.produced}
    assert topics == {LABELS_TOPIC}
    _, first_key, first_value = producer.produced[0]
    assert first_key == b"101"
    assert deserialize_label(first_value).is_fraud == 0


def test_shutdown_before_start_publishes_nothing(tmp_path: Path) -> None:
    source = tmp_path / "test.parquet"
    _frame().to_parquet(source)
    producer = _FakeProducer()
    shutdown = _ShutdownFlag()
    shutdown.requested = True

    published = run_label_simulator(
        _cfg(),
        source,
        lead_seconds=10.0,
        producer=producer,  # type: ignore[arg-type]
        shutdown=shutdown,
    )

    assert published == 0
    assert producer.produced == []
