from confluent_kafka import KafkaException

from fraud.serving.inference_log import InferenceLogger
from fraud.streaming.events import ScoredFeaturesEvent, deserialize_scored_features


def _event() -> ScoredFeaturesEvent:
    return ScoredFeaturesEvent(
        transaction_id="t-1",
        model_version=5,
        fraud_score=0.3,
        decision=False,
        features={"amt_log": 2.5, "TransactionAmt": 12.0},
    )


class _FakeProducer:
    def __init__(
        self, raise_error: Exception | None = None, delivery_error: object | None = None
    ) -> None:
        self.produced: list[tuple[str, bytes, bytes]] = []
        self.polls = 0
        self.flushes = 0
        self._raise_error = raise_error
        self._delivery_error = delivery_error

    def produce(self, topic: str, key: bytes, value: bytes, on_delivery: object = None) -> None:
        if self._raise_error is not None:
            raise self._raise_error
        self.produced.append((topic, key, value))
        if callable(on_delivery):
            on_delivery(self._delivery_error, None)

    def poll(self, _timeout: float) -> int:
        self.polls += 1
        return 0

    def flush(self, _timeout: float) -> int:
        self.flushes += 1
        return 0


def test_log_publishes_serialized_event_keyed_by_transaction() -> None:
    producer = _FakeProducer()
    logger = InferenceLogger(producer, topic="scored-features")  # type: ignore[arg-type]
    logger.log(_event())

    assert len(producer.produced) == 1
    topic, key, value = producer.produced[0]
    assert topic == "scored-features"
    assert key == b"t-1"
    assert deserialize_scored_features(value) == _event()
    assert producer.polls == 1


def test_log_swallows_buffer_error_without_raising() -> None:
    producer = _FakeProducer(raise_error=BufferError("queue full"))
    logger = InferenceLogger(producer)  # type: ignore[arg-type]
    logger.log(_event())  # must not raise
    assert producer.produced == []
    assert logger._dropped == 1


def test_log_swallows_kafka_exception_without_raising() -> None:
    producer = _FakeProducer(raise_error=KafkaException("broker unavailable"))
    logger = InferenceLogger(producer)  # type: ignore[arg-type]
    logger.log(_event())  # must not raise
    assert producer.produced == []
    assert logger._dropped == 1


def test_delivery_error_counts_as_dropped() -> None:
    producer = _FakeProducer(delivery_error="broker rejected")
    logger = InferenceLogger(producer)  # type: ignore[arg-type]
    logger.log(_event())
    assert logger._dropped == 1


def test_successful_delivery_does_not_count_as_dropped() -> None:
    producer = _FakeProducer()
    logger = InferenceLogger(producer)  # type: ignore[arg-type]
    logger.log(_event())
    assert logger._dropped == 0


def test_close_flushes() -> None:
    producer = _FakeProducer()
    InferenceLogger(producer).close()  # type: ignore[arg-type]
    assert producer.flushes == 1
