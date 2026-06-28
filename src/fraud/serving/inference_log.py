"""Fire-and-forget logging of scored features for downstream drift monitoring."""

from __future__ import annotations

from confluent_kafka import KafkaException, Producer

from fraud.common.logging import get_logger
from fraud.streaming.events import SCORED_FEATURES_TOPIC, ScoredFeaturesEvent, serialize

log = get_logger(__name__)

_DROP_LOG_EVERY = 1000


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
                self._topic, key=event.transaction_id.encode("utf-8"), value=serialize(event)
            )
            self._producer.poll(0)
        except (BufferError, KafkaException):
            self._dropped += 1
            if self._dropped % _DROP_LOG_EVERY == 1:
                log.warning("inference_log_dropped", dropped=self._dropped)

    def close(self) -> None:
        self._producer.flush(5.0)
