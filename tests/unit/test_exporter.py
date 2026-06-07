import dataclasses
import math
from typing import Any

import pandas as pd
from confluent_kafka import KafkaException

from fraud.ingestion.stream import ScoredFeaturesEvent, serialize
from fraud.monitoring.config import MonitoringConfig
from fraud.monitoring.drift import FeatureDrift
from fraud.monitoring.exporter import (
    DRIFT_KIND_DATA,
    DRIFT_KIND_PERF,
    BreachLatch,
    MonitorState,
    _commit,
    _route,
)
from fraud.training.features import FEATURE_COLUMNS


def test_breach_latch_fires_once_then_rearms_on_recovery() -> None:
    latch = BreachLatch(required=2)
    assert latch.update(True) is False
    assert latch.update(True) is True
    assert latch.update(True) is False
    assert latch.update(False) is False
    assert latch.update(True) is False
    assert latch.update(True) is True


class _FakeProducer:
    def __init__(self) -> None:
        self.produced: list[tuple[str, bytes]] = []

    def produce(self, topic: str, value: bytes) -> None:
        self.produced.append((topic, value))

    def poll(self, _timeout: float) -> None:
        return None

    def flush(self, *_args: Any) -> None:
        return None


def _cfg(**overrides: Any) -> MonitoringConfig:
    base = MonitoringConfig.from_settings()
    return dataclasses.replace(base, **overrides)


def _baseline() -> pd.DataFrame:
    return pd.DataFrame({name: [0.0, 1.0] for name in FEATURE_COLUMNS})


def _scored(transaction_id: str) -> ScoredFeaturesEvent:
    return ScoredFeaturesEvent(
        transaction_id=transaction_id,
        model_version=5,
        fraud_score=0.5,
        decision=False,
        features=dict.fromkeys(FEATURE_COLUMNS, 0.5),
    )


def _drift_drifted(
    _ref: pd.DataFrame, _cur: pd.DataFrame, _cols: Any, *, psi_threshold: float
) -> FeatureDrift:
    return FeatureDrift(psi={"amt_log": 0.9}, psi_threshold=psi_threshold)


def _drift_clean(
    _ref: pd.DataFrame, _cur: pd.DataFrame, _cols: Any, *, psi_threshold: float
) -> FeatureDrift:
    return FeatureDrift(psi={"amt_log": 0.01}, psi_threshold=psi_threshold)


def test_data_drift_alert_emitted_only_after_debounce() -> None:
    producer = _FakeProducer()
    cfg = _cfg(drift_debounce_cycles=2, min_current_for_drift=1)
    state = MonitorState(cfg, _baseline(), producer, drift_fn=_drift_drifted)  # type: ignore[arg-type]
    state.handle_scored_features(_scored("t-1"))

    state.recompute()
    assert producer.produced == []
    state.recompute()

    assert len(producer.produced) == 1
    topic, _ = producer.produced[0]
    assert topic == cfg.drift_alerts_topic


def test_clean_distribution_emits_no_alert() -> None:
    producer = _FakeProducer()
    cfg = _cfg(drift_debounce_cycles=1, min_current_for_drift=1)
    state = MonitorState(cfg, _baseline(), producer, drift_fn=_drift_clean)  # type: ignore[arg-type]
    state.handle_scored_features(_scored("t-1"))
    state.recompute()
    state.recompute()
    assert producer.produced == []


def test_perf_decay_alert_emitted_when_auprc_below_floor() -> None:
    producer = _FakeProducer()
    cfg = _cfg(
        drift_debounce_cycles=1,
        min_current_for_drift=10_000,
        min_matched_for_auprc=2,
        auprc_floor=2.0,
    )
    state = MonitorState(cfg, _baseline(), producer, drift_fn=_drift_clean)  # type: ignore[arg-type]
    for i in range(4):
        tid = f"t-{i}"
        state.handle_scored_features(_scored(tid))
        state.handle_label(tid, i % 2)
    state.recompute()

    assert len(producer.produced) == 1


def test_null_feature_is_recorded_as_nan_in_window() -> None:
    producer = _FakeProducer()
    cfg = _cfg(min_current_for_drift=1)
    state = MonitorState(cfg, _baseline(), producer, drift_fn=_drift_clean)  # type: ignore[arg-type]
    event = ScoredFeaturesEvent(
        transaction_id="t-1",
        model_version=5,
        fraud_score=0.5,
        decision=False,
        features={**dict.fromkeys(FEATURE_COLUMNS, 0.5), "C1": None},
    )

    state.handle_scored_features(event)

    window_row = state._features[-1]
    assert math.isnan(window_row["C1"])
    assert window_row["amt_log"] == 0.5


def test_route_skips_poison_message_without_raising() -> None:
    producer = _FakeProducer()
    cfg = _cfg(min_current_for_drift=1)
    state = MonitorState(cfg, _baseline(), producer, drift_fn=_drift_clean)  # type: ignore[arg-type]

    class _Msg:
        def value(self) -> bytes:
            return b"not-json"

        def topic(self) -> str:
            return cfg.scored_features_topic

    _route(state, cfg, _Msg())  # type: ignore[arg-type]
    # A valid scored-features message is handled.
    valid = serialize(_scored("t-1"))

    class _Good:
        def value(self) -> bytes:
            return valid

        def topic(self) -> str:
            return cfg.scored_features_topic

    _route(state, cfg, _Good())  # type: ignore[arg-type]
    state.recompute()
    assert producer.produced == []


def test_alert_kinds_are_distinct() -> None:
    assert DRIFT_KIND_DATA != DRIFT_KIND_PERF


class _CommitConsumer:
    def __init__(self, error: Exception | None = None) -> None:
        self.commits = 0
        self._error = error

    def commit(self, asynchronous: bool) -> None:
        self.commits += 1
        if self._error is not None:
            raise self._error


def test_commit_skips_when_nothing_consumed() -> None:
    consumer = _CommitConsumer()
    _commit(consumer, 0)  # type: ignore[arg-type]
    assert consumer.commits == 0


def test_commit_runs_when_messages_consumed() -> None:
    consumer = _CommitConsumer()
    _commit(consumer, 5)  # type: ignore[arg-type]
    assert consumer.commits == 1


def test_commit_swallows_kafka_errors() -> None:
    consumer = _CommitConsumer(error=KafkaException("no offset"))
    _commit(consumer, 5)  # type: ignore[arg-type]  # must not raise
    assert consumer.commits == 1
