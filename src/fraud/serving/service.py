from __future__ import annotations

import time

import bentoml
from pydantic import BaseModel, Field

from fraud.common.logging import configure_logging, get_logger
from fraud.config import get_settings
from fraud.serving.config import ServingConfig
from fraud.serving.features import OnlineFeatureFetcher, redis_reachable
from fraud.serving.model import load_champion
from fraud.serving.predict import score_transaction

log = get_logger(__name__)

MILLISECONDS_PER_SECOND = 1000.0


class PredictRequest(BaseModel):
    card_id: str = Field(min_length=1, max_length=128)
    amount: float = Field(gt=0.0)
    transaction_id: str | None = Field(default=None, max_length=128)


class PredictResponse(BaseModel):
    transaction_id: str | None
    fraud_score: float
    decision: bool
    threshold: float
    model_version: int
    latency_ms: float


@bentoml.service(name="argus_fraud_serving")
class FraudService:
    def __init__(self) -> None:
        settings = get_settings()
        configure_logging(settings.log_level, settings.log_json)
        self._cfg = ServingConfig.from_settings()
        self._bundle = load_champion(self._cfg)
        self._fetcher = OnlineFeatureFetcher(self._cfg)
        log.info(
            "model_loaded",
            version=self._bundle.version,
            family=self._bundle.family,
            threshold=self._bundle.threshold,
        )

    @bentoml.api  # type: ignore[untyped-decorator]
    def predict(self, request: PredictRequest) -> PredictResponse:
        start = time.perf_counter()
        features = self._fetcher.fetch(request.card_id, request.amount)
        scored = score_transaction(self._bundle, features)
        latency_ms = (time.perf_counter() - start) * MILLISECONDS_PER_SECOND
        log.info(
            "prediction_served",
            transaction_id=request.transaction_id,
            decision=scored.decision,
            fraud_score=scored.fraud_score,
            latency_ms=latency_ms,
        )
        return PredictResponse(
            transaction_id=request.transaction_id,
            fraud_score=scored.fraud_score,
            decision=scored.decision,
            threshold=scored.threshold,
            model_version=self._bundle.version,
            latency_ms=latency_ms,
        )

    def __is_alive__(self) -> bool:
        return True

    def __is_ready__(self) -> bool:
        ready = redis_reachable(self._cfg)
        if not ready:
            log.warning("redis_unavailable", connection=self._cfg.redis_connection)
        return ready
