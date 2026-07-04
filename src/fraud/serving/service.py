"""BentoML serving app: the authenticated /predict endpoint and its readiness checks."""

from __future__ import annotations

import time

import bentoml
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from fraud.common.logging import configure_logging, get_logger
from fraud.config import get_settings
from fraud.model_loader import ModelBundle, load_champion
from fraud.registry import get_alias_version
from fraud.serving.config import ServingConfig
from fraud.serving.features import OnlineFeatureFetcher, redis_reachable
from fraud.serving.inference_log import InferenceLogger
from fraud.serving.predict import score_transaction
from fraud.serving.reload import ChampionReloader
from fraud.serving.security import (
    MAX_REQUEST_BYTES,
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    verify_api_key,
)
from fraud.streaming.events import (
    MAX_TRANSACTION_AMOUNT,
    PredictResponse,
    RawAttributes,
    ScoredFeaturesEvent,
)

log = get_logger(__name__)

MILLISECONDS_PER_SECOND = 1000.0
# Hard cap on in-flight requests per replica: excess is shed with 429 instead of piling up and
# exhausting the worker. Horizontal scale, not this number, carries real throughput.
MAX_CONCURRENT_REQUESTS = 64
# A request this far over the 50ms SLO is wedged; time it out so it frees its concurrency slot
# instead of holding it for BentoML's 60s default. Kept below the consumer's 5s client timeout
# so a slow request returns a 504 the caller can classify, not a client-side connection error.
MAX_REQUEST_DURATION_SECONDS = 4


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    card_id: str = Field(min_length=1, max_length=128)
    amount: float = Field(gt=0.0, le=MAX_TRANSACTION_AMOUNT)
    transaction_id: str | None = Field(default=None, min_length=1, max_length=128)
    raw: RawAttributes = Field(default_factory=RawAttributes)


class _Runtime:
    __slots__ = ("bundle", "fetcher")

    def __init__(self, bundle: ModelBundle, fetcher: OnlineFeatureFetcher) -> None:
        self.bundle = bundle
        self.fetcher = fetcher


def _resolve_api_key(api_key: SecretStr | None, environment: str) -> str | None:
    if api_key is not None:
        return api_key.get_secret_value()
    if environment != "local":
        raise RuntimeError(
            "serving_api_key must be set outside local; refusing to start an open predict API"
        )
    log.warning("serving_auth_disabled")
    return None


@bentoml.service(
    name="argus_fraud_serving",
    traffic={
        "max_concurrency": MAX_CONCURRENT_REQUESTS,
        "timeout": MAX_REQUEST_DURATION_SECONDS,
    },
)
class FraudService:
    def __init__(self) -> None:
        settings = get_settings()
        configure_logging(settings.log_level, settings.log_json)
        self._cfg = ServingConfig.from_settings()
        self._api_key = _resolve_api_key(settings.serving_api_key, settings.environment)
        self._runtime = self._build_runtime(load_champion(self._cfg.champion_load_config))
        self._inference_log = InferenceLogger.from_bootstrap(settings.kafka_bootstrap_servers)
        log.info(
            "model_loaded",
            version=self._runtime.bundle.version,
            family=self._runtime.bundle.family,
            threshold=self._runtime.bundle.threshold,
        )
        self._reloader = ChampionReloader(
            interval_seconds=self._cfg.reload_interval_seconds,
            current_version=lambda: self._runtime.bundle.version,
            latest_version=lambda: get_alias_version(
                self._cfg.model_name, self._cfg.champion_alias
            ),
            load=lambda: load_champion(self._cfg.champion_load_config),
            apply=self._apply_bundle,
        )
        self._reloader.start()

    def _build_runtime(self, bundle: ModelBundle) -> _Runtime:
        return _Runtime(bundle, OnlineFeatureFetcher(self._cfg, bundle.encoder))

    def _apply_bundle(self, bundle: ModelBundle) -> None:
        self._runtime = self._build_runtime(bundle)

    @bentoml.api  # type: ignore[untyped-decorator]
    def predict(self, request: PredictRequest, ctx: bentoml.Context) -> PredictResponse:
        self._authenticate(ctx)
        runtime = self._runtime
        start = time.perf_counter()
        features = runtime.fetcher.fetch(request.card_id, request.amount, request.raw)
        scored = score_transaction(runtime.bundle, features)
        latency_ms = (time.perf_counter() - start) * MILLISECONDS_PER_SECOND
        self._log_inference(
            request.transaction_id,
            runtime.bundle.version,
            features,
            scored.fraud_score,
            scored.decision,
        )
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
            model_version=runtime.bundle.version,
            latency_ms=latency_ms,
        )

    def _authenticate(self, ctx: bentoml.Context) -> None:
        if self._api_key is None:
            return
        verify_api_key(ctx.request.headers.get("authorization"), self._api_key)

    def _log_inference(
        self,
        transaction_id: str | None,
        model_version: int,
        features: pd.DataFrame,
        fraud_score: float,
        decision: bool,
    ) -> None:
        if transaction_id is None:
            return
        try:
            self._inference_log.log(
                ScoredFeaturesEvent(
                    transaction_id=transaction_id,
                    model_version=model_version,
                    fraud_score=fraud_score,
                    decision=decision,
                    features={
                        str(name): None if pd.isna(value) else float(value)
                        for name, value in features.iloc[0].items()
                    },
                )
            )
        except Exception as exc:
            log.warning("inference_log_failed", error=str(exc), exc_info=True)

    @bentoml.on_shutdown  # type: ignore[untyped-decorator]
    def _shutdown(self) -> None:
        self._reloader.stop()
        self._inference_log.close()

    def __is_alive__(self) -> bool:
        return True

    def __is_ready__(self) -> bool:
        ready = redis_reachable(self._cfg)
        if not ready:
            log.warning("redis_unavailable", connection=self._cfg.redis_connection)
        return ready


FraudService.add_asgi_middleware(RateLimitMiddleware)  # type: ignore[attr-defined]
FraudService.add_asgi_middleware(BodySizeLimitMiddleware, max_bytes=MAX_REQUEST_BYTES)  # type: ignore[attr-defined]
