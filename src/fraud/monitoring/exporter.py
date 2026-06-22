"""Prometheus exporter that turns the prediction and label streams into ML health metrics."""

from __future__ import annotations

import math
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import FrameType
from typing import Protocol

import pandas as pd
from confluent_kafka import Consumer, KafkaException, Message, Producer
from prometheus_client import Counter, Gauge, start_http_server
from pydantic import ValidationError

from fraud.common.logging import configure_logging, get_logger
from fraud.config import get_settings
from fraud.ingestion.stream import (
    DriftAlertEvent,
    ScoredFeaturesEvent,
    deserialize_label,
    deserialize_scored_features,
    serialize,
)
from fraud.monitoring.baseline import load_baseline
from fraud.monitoring.config import MonitoringConfig
from fraud.monitoring.drift import FeatureDrift, compute_feature_drift
from fraud.monitoring.perf_monitor import RollingPerformance
from fraud.training.features import FEATURE_COLUMNS

log = get_logger(__name__)

POLL_TIMEOUT_SECONDS = 1.0

DRIFT_KIND_DATA = "data_drift"
DRIFT_KIND_PERF = "concept_drift"

ROLLING_AUPRC = Gauge("argus_rolling_auprc", "Rolling AUPRC over labeled predictions")
BUSINESS_COST = Gauge("argus_business_cost_per_txn_usd", "Realized cost per labeled transaction")
FLAGGED_RATE = Gauge("argus_flagged_rate", "Share of labeled transactions flagged as fraud")
# The top-N PSI cap relies on clear(), so the monitor must run as a single process.
FEATURE_PSI = Gauge("argus_feature_drift_psi", "Feature PSI vs training baseline", ["feature"])
FEATURE_PSI_MAX = Gauge("argus_feature_drift_psi_max", "Highest feature PSI vs training baseline")
DRIFTED_FEATURES = Gauge("argus_drifted_features", "Model-input features with PSI above threshold")
MATCHED_TOTAL = Gauge("argus_matched_join", "Predictions joined with a label in the window")
PENDING_JOIN = Gauge("argus_pending_join", "Predictions awaiting their delayed label")
MODEL_VERSION = Gauge("argus_monitored_model_version", "Champion version under monitoring")
SCORED_EVENTS = Counter("argus_scored_events_total", "Scored-features events consumed")
LABEL_EVENTS = Counter("argus_label_events_total", "Label events consumed")
DRIFT_ALERTS = Counter("argus_drift_alerts_total", "Drift alerts published", ["kind"])


def _feature_value(value: float | None) -> float:
    return math.nan if value is None else value


def _top_psi(psi: dict[str, float], top_n: int) -> list[tuple[str, float]]:
    return sorted(psi.items(), key=lambda item: item[1], reverse=True)[:top_n]


class DriftFn(Protocol):
    def __call__(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        columns: tuple[str, ...],
        *,
        psi_threshold: float,
    ) -> FeatureDrift: ...


@dataclass(slots=True)
class BreachLatch:
    """Fires once after a breach holds for `required` consecutive checks; rearms on recovery."""

    required: int
    _streak: int = 0
    _fired: bool = False

    def update(self, breached: bool) -> bool:
        if not breached:
            self._streak = 0
            self._fired = False
            return False
        self._streak += 1
        if self._streak >= self.required and not self._fired:
            self._fired = True
            return True
        return False


@dataclass(slots=True)
class MonitorState:
    cfg: MonitoringConfig
    baseline: pd.DataFrame
    producer: Producer
    drift_fn: DriftFn = compute_feature_drift
    _perf: RollingPerformance = field(init=False)
    _features: deque[dict[str, float]] = field(init=False)
    _data_latch: BreachLatch = field(init=False)
    _perf_latch: BreachLatch = field(init=False)

    def __post_init__(self) -> None:
        self._perf = RollingPerformance(
            cost_matrix=self.cfg.cost_matrix,
            window_size=self.cfg.window_size,
            join_retention=self.cfg.join_retention,
        )
        self._features = deque(maxlen=self.cfg.window_size)
        self._data_latch = BreachLatch(self.cfg.drift_debounce_cycles)
        self._perf_latch = BreachLatch(self.cfg.drift_debounce_cycles)

    def handle_scored_features(self, event: ScoredFeaturesEvent) -> None:
        self._perf.observe_score(event.transaction_id, event.fraud_score, event.decision)
        # null is a missing feature; keep it NaN to match the baseline, not a fake zero.
        self._features.append(
            {name: _feature_value(event.features.get(name)) for name in FEATURE_COLUMNS}
        )
        SCORED_EVENTS.inc()
        MODEL_VERSION.set(event.model_version)

    def handle_label(self, transaction_id: str, is_fraud: int) -> None:
        self._perf.observe_label(transaction_id, is_fraud)
        LABEL_EVENTS.inc()

    def recompute(self) -> None:
        drift = self._compute_drift()
        for alert in self._update_metrics(drift):
            self._emit(alert)

    def _compute_drift(self) -> FeatureDrift | None:
        if len(self._features) < self.cfg.min_current_for_drift:
            return None
        current = pd.DataFrame(list(self._features), columns=list(FEATURE_COLUMNS))
        try:
            return self.drift_fn(
                self.baseline, current, FEATURE_COLUMNS, psi_threshold=self.cfg.psi_threshold
            )
        except ValueError as exc:
            # A degenerate window (e.g. an all-null feature) must not crash the monitor; skip
            # this drift cycle. Rolling AUPRC and cost still update.
            log.warning("drift_computation_skipped", error=str(exc))
            return None

    def _update_metrics(self, drift: FeatureDrift | None) -> list[DriftAlertEvent]:
        auprc = self._perf.rolling_auprc()
        ROLLING_AUPRC.set(auprc)
        BUSINESS_COST.set(self._perf.business_cost_per_txn())
        FLAGGED_RATE.set(self._perf.flagged_rate())
        MATCHED_TOTAL.set(self._perf.matched_count)
        PENDING_JOIN.set(self._perf.pending_count)

        alerts: list[DriftAlertEvent] = []
        if drift is not None:
            self._publish_psi(drift)
            if self._data_latch.update(bool(drift.drifted_features)):
                alerts.append(_data_drift_alert(drift))

        if self._perf_latch.update(self._auprc_below_floor(auprc)):
            alerts.append(_perf_decay_alert(auprc, self.cfg.auprc_floor))
        return alerts

    def _publish_psi(self, drift: FeatureDrift) -> None:
        # Clear first so a feature dropping out of the top-N stops reporting a stale series.
        FEATURE_PSI.clear()
        for feature, psi in _top_psi(drift.psi, self.cfg.psi_top_n):
            FEATURE_PSI.labels(feature=feature).set(psi)
        FEATURE_PSI_MAX.set(drift.max_psi)
        DRIFTED_FEATURES.set(len(drift.drifted_features))

    def _auprc_below_floor(self, auprc: float) -> bool:
        if math.isnan(auprc) or self._perf.matched_count < self.cfg.min_matched_for_auprc:
            return False
        return auprc < self.cfg.auprc_floor

    def _emit(self, alert: DriftAlertEvent) -> None:
        self.producer.produce(self.cfg.drift_alerts_topic, value=serialize(alert))
        self.producer.poll(0)
        DRIFT_ALERTS.labels(kind=alert.kind).inc()
        log.warning(
            "drift_alert_published",
            kind=alert.kind,
            metric=alert.metric,
            value=alert.value,
            threshold=alert.threshold,
        )


def _data_drift_alert(drift: FeatureDrift) -> DriftAlertEvent:
    return DriftAlertEvent(
        kind=DRIFT_KIND_DATA,
        metric="feature_drift_psi",
        value=drift.max_psi,
        threshold=drift.psi_threshold,
        detected_at=_now(),
    )


def _perf_decay_alert(auprc: float, floor: float) -> DriftAlertEvent:
    return DriftAlertEvent(
        kind=DRIFT_KIND_PERF,
        metric="rolling_auprc",
        value=auprc,
        threshold=floor,
        detected_at=_now(),
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()


class _ShutdownFlag:
    def __init__(self) -> None:
        self.requested = False

    def request(self, _signum: int, _frame: FrameType | None) -> None:
        self.requested = True


def run_exporter(cfg: MonitoringConfig, shutdown: _ShutdownFlag | None = None) -> None:
    """Consume scored-features and labels, expose ML health metrics, alert on drift."""
    shutdown = shutdown or _install_shutdown_handler()
    state = MonitorState(
        cfg, load_baseline(cfg), Producer({"bootstrap.servers": cfg.bootstrap_servers})
    )
    consumer = _build_consumer(cfg)
    consumer.subscribe([cfg.scored_features_topic, cfg.labels_topic])
    start_http_server(cfg.exporter_port)
    log.info("exporter_start", port=cfg.exporter_port, baseline_rows=len(state.baseline))

    last_recompute = time.monotonic()
    consumed = 0
    try:
        while not shutdown.requested:
            message = consumer.poll(POLL_TIMEOUT_SECONDS)
            if message is not None and message.error() is None:
                _route(state, cfg, message)
                consumed += 1
            if time.monotonic() - last_recompute >= cfg.recompute_interval_seconds:
                state.recompute()
                _commit(consumer, consumed)
                consumed = 0
                last_recompute = time.monotonic()
    finally:
        consumer.close()
        state.producer.flush()
        log.info("exporter_stopped")


def _commit(consumer: Consumer, consumed: int) -> None:
    # Nothing is stored to commit until a message has been consumed; a transient
    # broker hiccup on commit must not take down a read-side monitor.
    if consumed == 0:
        return
    try:
        consumer.commit(asynchronous=False)
    except KafkaException as exc:
        log.warning("monitor_commit_failed", error=str(exc))


def _route(state: MonitorState, cfg: MonitoringConfig, message: Message) -> None:
    payload = message.value()
    if payload is None:
        return
    try:
        if message.topic() == cfg.scored_features_topic:
            state.handle_scored_features(deserialize_scored_features(payload))
        else:
            label = deserialize_label(payload)
            state.handle_label(label.transaction_id, label.is_fraud)
    except ValidationError as exc:
        log.warning("monitor_message_skipped", topic=message.topic(), error=str(exc))


def _build_consumer(cfg: MonitoringConfig) -> Consumer:
    return Consumer(
        {
            "bootstrap.servers": cfg.bootstrap_servers,
            "group.id": cfg.consumer_group,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
        }
    )


def _install_shutdown_handler() -> _ShutdownFlag:
    shutdown = _ShutdownFlag()
    signal.signal(signal.SIGINT, shutdown.request)
    signal.signal(signal.SIGTERM, shutdown.request)
    return shutdown


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    run_exporter(MonitoringConfig.from_settings())


if __name__ == "__main__":
    main()
