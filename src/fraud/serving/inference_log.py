"""Fire-and-forget logging of scored features for downstream drift monitoring."""

from __future__ import annotations

from confluent_kafka import KafkaException, Producer
from prometheus_client import Counter

from fraud.common.logging import get_logger
from fraud.streaming.events import SCORED_FEATURES_TOPIC, ScoredFeaturesEvent, serialize

log = get_logger(__name__)

_DROP_LOG_EVERY = 1000

INFERENCE_LOG_DROPPED = Counter(
    "argus_inference_log_dropped_total", "Scored-features records dropped before reaching Kafka"
)


class InferenceLogger:
    """Publishes the exact scored features without ever blocking or failing predict.

    A full local queue or an unreachable broker drops the record and keeps
    serving, so monitoring can never degrade the inference path.
    """

    def __init__(self, producer: Producer, topic: str = SCORED_FEATURES_TOPIC) -> None:
        self._producer = producer
        self._topic = topic
        self._dropped = 0

    @classmethod
    def from_bootstrap(
        cls, bootstrap_servers: str, topic: str = SCORED_FEATURES_TOPIC
    ) -> InferenceLogger:
        # Expire undeliverable records fast and cap the local queue so a broker outage
        # drops logs instead of growing serving's memory.
        return cls(
            Producer(
                {
                    "bootstrap.servers": bootstrap_servers,
                    "message.timeout.ms": "5000",
                    "queue.buffering.max.messages": "10000",
                }
            ),
            topic,
        )

    def log(self, event: ScoredFeaturesEvent) -> None:
        try:
            self._producer.produce(
                self._topic,
                key=event.transaction_id.encode("utf-8"),
                value=serialize(event),
                on_delivery=self._on_delivery,
            )
            self._producer.poll(0)
        except (BufferError, KafkaException):
            self._note_dropped()

    def _on_delivery(self, error: object, _message: object) -> None:
        if error is not None:
            self._note_dropped()

    def _note_dropped(self) -> None:
        self._dropped += 1
        INFERENCE_LOG_DROPPED.inc()
        if self._dropped % _DROP_LOG_EVERY == 1:
            log.warning("inference_log_dropped", dropped=self._dropped)

    def close(self) -> None:
        pending = self._producer.flush(5.0)
        if pending:
            log.warning("inference_log_close_dropped", pending=pending)
