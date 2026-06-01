from fraud.ingestion.stream import ScoredFeaturesEvent, deserialize_scored_features
from fraud.serving.inference_log import InferenceLogger


def _event() -> ScoredFeaturesEvent:
    return ScoredFeaturesEvent(
        transaction_id="t-1",
        model_version=5,
        fraud_score=0.3,
        decision=False,
        features={"amt_log": 2.5, "TransactionAmt": 12.0},
    )


class _FakeProducer:
    def __init__(self, raise_error: Exception | None = None) -> None:
        self.produced: list[tuple[str, bytes, bytes]] = []
        self.polls = 0
        self.flushes = 0
        self._raise_error = raise_error

    def produce(self, topic: str, key: bytes, value: bytes) -> None:
        if self._raise_error is not None:
            raise self._raise_error
        self.produced.append((topic, key, value))

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


def test_close_flushes() -> None:
    producer = _FakeProducer()
    InferenceLogger(producer).close()  # type: ignore[arg-type]
    assert producer.flushes == 1
