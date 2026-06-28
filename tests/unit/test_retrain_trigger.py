from typing import cast

from confluent_kafka import Message

from fraud.ingestion.retrain_trigger import (
    CooldownGate,
    RetrainTriggerConfig,
    _handle_alert,
)
from fraud.streaming.events import (
    DRIFT_ALERTS_TOPIC,
    RETRAIN_GROUP,
    DriftAlertEvent,
    serialize,
)


class _FakeMessage:
    def __init__(self, payload: bytes | None) -> None:
        self._payload = payload

    def value(self) -> bytes | None:
        return self._payload


def _alert_bytes() -> bytes:
    return serialize(
        DriftAlertEvent(
            kind="data_drift",
            metric="feature_drift_psi",
            value=0.4,
            threshold=0.2,
            detected_at="2017-12-01T00:00:00+00:00",
        )
    )


def _drive(payload: bytes | None, gate: CooldownGate, now: float) -> list[DriftAlertEvent]:
    fired: list[DriftAlertEvent] = []
    message = cast(Message, _FakeMessage(payload))
    _handle_alert(message, gate, fired.append, lambda: now)
    return fired


def test_cooldown_gate_collapses_burst_then_reopens() -> None:
    gate = CooldownGate(cooldown_seconds=100.0)
    assert gate.should_fire(0.0)
    assert not gate.should_fire(50.0)
    assert gate.should_fire(100.0)


def test_handle_alert_fires_trigger_for_valid_alert() -> None:
    fired = _drive(_alert_bytes(), CooldownGate(0.0), 0.0)
    assert len(fired) == 1
    assert fired[0].kind == "data_drift"


def test_handle_alert_skips_poison_message() -> None:
    assert _drive(b'{"kind": "x"}', CooldownGate(0.0), 0.0) == []


def test_handle_alert_skips_empty_payload() -> None:
    assert _drive(None, CooldownGate(0.0), 0.0) == []


def test_handle_alert_honors_cooldown_across_calls() -> None:
    gate = CooldownGate(100.0)
    first = _drive(_alert_bytes(), gate, 0.0)
    second = _drive(_alert_bytes(), gate, 10.0)
    assert len(first) == 1
    assert second == []


def test_config_from_settings_binds_drift_topic_and_retrain_group() -> None:
    cfg = RetrainTriggerConfig.from_settings()
    assert cfg.drift_alerts_topic == DRIFT_ALERTS_TOPIC
    assert cfg.consumer_group == RETRAIN_GROUP
    assert cfg.deployment_name
