"""Bridge the drift-alerts topic to the Prefect retraining deployment."""

from __future__ import annotations

import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import FrameType

from confluent_kafka import Consumer, Message
from pydantic import ValidationError

from fraud.common.logging import configure_logging, get_logger
from fraud.config import get_settings
from fraud.params import load_params
from fraud.streaming.events import (
    DRIFT_ALERTS_TOPIC,
    RETRAIN_GROUP,
    DriftAlertEvent,
    deserialize_drift_alert,
)

log = get_logger(__name__)

POLL_TIMEOUT_SECONDS = 1.0


@dataclass(slots=True)
class CooldownGate:
    """Collapses a burst of drift alerts into one retrain per cooldown window."""

    cooldown_seconds: float
    last_fired: float | None = None

    def should_fire(self, now: float) -> bool:
        if self.last_fired is not None and now - self.last_fired < self.cooldown_seconds:
            return False
        self.last_fired = now
        return True


@dataclass(frozen=True, slots=True)
class RetrainTriggerConfig:
    bootstrap_servers: str
    drift_alerts_topic: str
    consumer_group: str
    deployment_name: str
    cooldown_seconds: float

    @classmethod
    def from_settings(cls) -> RetrainTriggerConfig:
        settings = get_settings()
        retraining = load_params().retraining
        return cls(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            drift_alerts_topic=DRIFT_ALERTS_TOPIC,
            consumer_group=RETRAIN_GROUP,
            deployment_name=retraining.deployment_name,
            cooldown_seconds=retraining.cooldown_seconds,
        )


TriggerFn = Callable[[DriftAlertEvent], None]


class _ShutdownFlag:
    def __init__(self) -> None:
        self.requested = False

    def request(self, _signum: int, _frame: FrameType | None) -> None:
        self.requested = True


def run_retrain_trigger(
    cfg: RetrainTriggerConfig,
    *,
    trigger: TriggerFn | None = None,
    clock: Callable[[], float] = time.monotonic,
    shutdown: _ShutdownFlag | None = None,
) -> None:
    """Fire the retraining deployment on each fresh drift alert, throttled by cooldown."""
    trigger = trigger or _deployment_trigger(cfg)
    gate = CooldownGate(cfg.cooldown_seconds)
    shutdown = shutdown or _install_shutdown_handler()
    consumer = _build_consumer(cfg)
    consumer.subscribe([cfg.drift_alerts_topic])
    log.info("retrain_trigger_start", topic=cfg.drift_alerts_topic, deployment=cfg.deployment_name)
    try:
        while not shutdown.requested:
            message = consumer.poll(POLL_TIMEOUT_SECONDS)
            if message is None:
                continue
            if message.error():
                log.warning("retrain_consume_error", error=str(message.error()))
                continue
            _handle_alert(message, gate, trigger, clock)
    finally:
        consumer.close()
        log.info("retrain_trigger_stopped")


def _handle_alert(
    message: Message, gate: CooldownGate, trigger: TriggerFn, clock: Callable[[], float]
) -> None:
    payload = message.value()
    if payload is None:
        return
    try:
        alert = deserialize_drift_alert(payload)
    except ValidationError as exc:
        log.warning("retrain_poison_message_skipped", error=str(exc))
        return
    if not gate.should_fire(clock()):
        log.info("retrain_skipped_cooldown", kind=alert.kind, metric=alert.metric)
        return
    log.info("retrain_fired", kind=alert.kind, metric=alert.metric, value=alert.value)
    trigger(alert)


def _build_consumer(cfg: RetrainTriggerConfig) -> Consumer:
    # Cooldown is the idempotency guard, so auto-commit at the latest offset is
    # enough: react to alerts that arrive while running, never replay stale ones.
    return Consumer(
        {
            "bootstrap.servers": cfg.bootstrap_servers,
            "group.id": cfg.consumer_group,
            "enable.auto.commit": True,
            "auto.offset.reset": "latest",
        }
    )


def _deployment_trigger(cfg: RetrainTriggerConfig) -> TriggerFn:
    from pipelines.flows.retraining_pipeline import RETRAIN_REASON_DRIFT
    from prefect.deployments import run_deployment

    def trigger(alert: DriftAlertEvent) -> None:
        try:
            run_deployment(
                name=cfg.deployment_name,
                parameters={"reason": f"{RETRAIN_REASON_DRIFT}:{alert.kind}"},
                timeout=0,
            )
        except Exception as exc:  # a Prefect API hiccup must not wedge the loop
            log.error("retrain_dispatch_failed", deployment=cfg.deployment_name, error=str(exc))

    return trigger


def _install_shutdown_handler() -> _ShutdownFlag:
    shutdown = _ShutdownFlag()
    signal.signal(signal.SIGINT, shutdown.request)
    signal.signal(signal.SIGTERM, shutdown.request)
    return shutdown


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    run_retrain_trigger(RetrainTriggerConfig.from_settings())


if __name__ == "__main__":
    main()
