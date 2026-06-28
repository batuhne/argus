import dataclasses
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
import pytest
from confluent_kafka import OFFSET_END, TopicPartition

from fraud.ingestion.stream import ScoredFeaturesEvent, serialize
from fraud.monitoring.config import MonitoringConfig
from fraud.monitoring.drift import FeatureDrift
from fraud.monitoring.exporter import (
    DRIFT_KIND_DATA,
    DRIFT_KIND_PERF,
    FEATURE_PSI,
    FEATURE_PSI_MAX,
    JOIN_CLOCK_LAG,
    LAST_RECOMPUTE,
    BreachLatch,
    MonitorState,
    _event_time_seconds,
    _route,
    _run_recompute_cycle,
    _top_psi,
    _window_start_partitions,
)
from fraud.transforms.features import FEATURE_COLUMNS


@pytest.fixture(autouse=True)
def _single_process_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    # The monitor runs single-process, so clear() bounds the PSI gauge. Drop any
    # multiprocess flag another test module set earlier so we exercise that path.
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    monkeypatch.delenv("prometheus_multiproc_dir", raising=False)


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


def _drift_raises(
    _ref: pd.DataFrame, _cur: pd.DataFrame, _cols: Any, *, psi_threshold: float
) -> FeatureDrift:
    raise ValueError("An empty column 'C1' was provided for drift calculation")


def _drift_ranked(
    _ref: pd.DataFrame, _cur: pd.DataFrame, _cols: Any, *, psi_threshold: float
) -> FeatureDrift:
    psi = {f"f{i}": round(0.9 - i * 0.1, 1) for i in range(6)}
    return FeatureDrift(psi=psi, psi_threshold=psi_threshold)


def test_data_drift_alert_emitted_only_after_debounce() -> None:
    producer = _FakeProducer()
    cfg = _cfg(drift_debounce_cycles=2, min_current_for_drift=1)
    state = MonitorState(cfg, _baseline(), producer, drift_fn=_drift_drifted)  # type: ignore[arg-type]
    state.handle_scored_features(_scored("t-1"), event_time=0.0)

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
    state.handle_scored_features(_scored("t-1"), event_time=0.0)
    state.recompute()
    state.recompute()
    assert producer.produced == []


def test_recompute_survives_drift_failure() -> None:
    producer = _FakeProducer()
    cfg = _cfg(drift_debounce_cycles=1, min_current_for_drift=1)
    state = MonitorState(cfg, _baseline(), producer, drift_fn=_drift_raises)  # type: ignore[arg-type]
    state.handle_scored_features(_scored("t-1"), event_time=0.0)

    # A degenerate window makes drift raise; recompute must not crash the monitor.
    state.recompute()

    assert producer.produced == []


def _gauge(metric: Any) -> float:
    return float(next(iter(metric.collect())).samples[0].value)


def test_update_fast_metrics_sets_join_clock_lag() -> None:
    cfg = _cfg(min_current_for_drift=10_000)
    state = MonitorState(cfg, _baseline(), _FakeProducer(), drift_fn=_drift_clean)  # type: ignore[arg-type]
    state.handle_scored_features(_scored("t-1"), event_time=time.time() - 100.0)

    state.update_fast_metrics()

    assert 90.0 <= _gauge(JOIN_CLOCK_LAG) <= 300.0


def test_join_clock_lag_untouched_without_events() -> None:
    cfg = _cfg(min_current_for_drift=10_000)
    state = MonitorState(cfg, _baseline(), _FakeProducer(), drift_fn=_drift_clean)  # type: ignore[arg-type]
    before = _gauge(JOIN_CLOCK_LAG)

    state.update_fast_metrics()  # no events, so the join clock is undefined and stays put

    assert _gauge(JOIN_CLOCK_LAG) == before


def test_update_fast_metrics_stamps_last_recompute() -> None:
    cfg = _cfg(min_current_for_drift=10_000)
    state = MonitorState(cfg, _baseline(), _FakeProducer(), drift_fn=_drift_clean)  # type: ignore[arg-type]

    state.update_fast_metrics()

    assert _gauge(LAST_RECOMPUTE) >= time.time() - 5.0


def test_recompute_cycle_offloads_drift_and_emits_when_ready() -> None:
    producer = _FakeProducer()
    cfg = _cfg(drift_debounce_cycles=1, min_current_for_drift=1)
    state = MonitorState(cfg, _baseline(), producer, drift_fn=_drift_drifted)  # type: ignore[arg-type]
    state.handle_scored_features(_scored("t-1"), event_time=0.0)

    with ThreadPoolExecutor(max_workers=1) as executor:
        job = _run_recompute_cycle(state, executor, None)
        assert job is not None
        assert producer.produced == []  # the drift result is not ready on the first cycle
        job.result()
        _run_recompute_cycle(state, executor, job)

    assert any(topic == cfg.drift_alerts_topic for topic, _ in producer.produced)


def test_top_psi_returns_highest_descending() -> None:
    assert _top_psi({"a": 0.1, "b": 0.9, "c": 0.4}, 2) == [("b", 0.9), ("c", 0.4)]


def test_top_psi_returns_all_when_fewer_than_requested() -> None:
    assert _top_psi({"a": 0.1}, 5) == [("a", 0.1)]


def test_top_psi_empty_returns_empty() -> None:
    assert _top_psi({}, 5) == []


def test_only_top_n_feature_psi_series_exposed() -> None:
    producer = _FakeProducer()
    cfg = _cfg(min_current_for_drift=1, psi_top_n=3)
    state = MonitorState(cfg, _baseline(), producer, drift_fn=_drift_ranked)  # type: ignore[arg-type]
    state.handle_scored_features(_scored("t-1"), event_time=0.0)

    state.recompute()

    exposed = {sample.labels["feature"] for sample in next(iter(FEATURE_PSI.collect())).samples}
    assert exposed == {"f0", "f1", "f2"}


def test_feature_psi_max_tracks_worst_feature() -> None:
    producer = _FakeProducer()
    cfg = _cfg(min_current_for_drift=1, psi_top_n=3)
    state = MonitorState(cfg, _baseline(), producer, drift_fn=_drift_ranked)  # type: ignore[arg-type]
    state.handle_scored_features(_scored("t-1"), event_time=0.0)

    state.recompute()

    assert next(iter(FEATURE_PSI_MAX.collect())).samples[0].value == 0.9


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
        state.handle_scored_features(_scored(tid), event_time=0.0)
        state.handle_label(tid, i % 2, event_time=0.0)
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

    state.handle_scored_features(event, event_time=0.0)

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

        def timestamp(self) -> tuple[int, int]:
            return (1, 0)

    _route(state, cfg, _Msg())  # type: ignore[arg-type]
    # A valid scored-features message is handled.
    valid = serialize(_scored("t-1"))

    class _Good:
        def value(self) -> bytes:
            return valid

        def topic(self) -> str:
            return cfg.scored_features_topic

        def timestamp(self) -> tuple[int, int]:
            return (1, 0)

    _route(state, cfg, _Good())  # type: ignore[arg-type]
    state.recompute()
    assert producer.produced == []


def test_alert_kinds_are_distinct() -> None:
    assert DRIFT_KIND_DATA != DRIFT_KIND_PERF


def test_event_time_uses_producer_timestamp_in_seconds() -> None:
    class _Ts:
        def timestamp(self) -> tuple[int, int]:
            return (1, 5000)

    assert _event_time_seconds(_Ts(), fallback=0.0) == 5.0  # type: ignore[arg-type]


def test_event_time_falls_back_to_join_clock_when_timestamp_missing() -> None:
    class _NoTs:
        def timestamp(self) -> tuple[int, int]:
            return (0, -1)

    # A timeless message must reuse the join clock, never jump to wall-clock now.
    assert _event_time_seconds(_NoTs(), fallback=123.0) == 123.0  # type: ignore[arg-type]


class _SeekConsumer:
    def __init__(self, located: list[TopicPartition]) -> None:
        self._located = located
        self.queried: list[TopicPartition] = []

    def offsets_for_times(
        self, partitions: list[TopicPartition], timeout: float
    ) -> list[TopicPartition]:
        self.queried = partitions
        return self._located


def test_window_start_uses_resolved_offsets_and_tails_when_too_recent() -> None:
    consumer = _SeekConsumer(
        located=[
            TopicPartition("scored-features", 0, 42),
            TopicPartition("labels", 0, -1),
        ]
    )
    requested = [TopicPartition("scored-features", 0), TopicPartition("labels", 0)]

    result = _window_start_partitions(consumer, requested, cutoff_ms=1000)  # type: ignore[arg-type]

    # cutoff written into each partition's offset for the lookup
    assert [tp.offset for tp in consumer.queried] == [1000, 1000]
    # resolved offset kept; a nothing-recent partition tails
    assert result[0].offset == 42
    assert result[1].offset == OFFSET_END
