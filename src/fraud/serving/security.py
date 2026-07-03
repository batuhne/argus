"""Request-time controls for the serving API: bearer auth, a body-size guard, and rate limiting."""

from __future__ import annotations

import hashlib
import hmac
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus

from bentoml.exceptions import BentoMLException
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Comfortably above a legit predict body (scalars plus the bounded v and categorical maps).
MAX_REQUEST_BYTES = 64 * 1024

# Caps one caller's sustained rate; max_concurrency only bounds global in-flight, not per-key.
RATE_LIMIT_CAPACITY = 120
RATE_LIMIT_REFILL_PER_SECOND = 20.0
# Cap the identity table so an unauthenticated flood of distinct clients cannot grow it without end.
MAX_TRACKED_CLIENTS = 10_000
# Kubelet probes and Prometheus scrapes must never be throttled, or readiness flaps under load.
UNLIMITED_PATHS = frozenset({"/livez", "/readyz", "/healthz", "/metrics"})


class Unauthorized(BentoMLException):
    error_code = HTTPStatus.UNAUTHORIZED


def verify_api_key(authorization: str | None, expected: str) -> None:
    """Raise Unauthorized unless the request carries the expected bearer token."""
    scheme, _, token = (authorization or "").partition(" ")
    provided = token.strip() if scheme.lower() == "bearer" else ""
    # Constant-time compare so a wrong key cannot be guessed byte by byte from response timing.
    if not provided or not hmac.compare_digest(provided, expected):
        raise Unauthorized("missing or invalid API key")


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    """Per-client token bucket, in-process per replica; scale out to raise the effective ceiling."""

    def __init__(
        self,
        *,
        capacity: int = RATE_LIMIT_CAPACITY,
        refill_per_second: float = RATE_LIMIT_REFILL_PER_SECOND,
        max_clients: int = MAX_TRACKED_CLIENTS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._capacity = float(capacity)
        self._refill = refill_per_second
        self._max_clients = max_clients
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def check(self, identity: str) -> tuple[bool, int]:
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(identity)
            if bucket is None:
                if len(self._buckets) >= self._max_clients:
                    self._evict_full()
                bucket = _Bucket(self._capacity, now)
                self._buckets[identity] = bucket
            bucket.tokens = min(
                self._capacity, bucket.tokens + (now - bucket.updated) * self._refill
            )
            bucket.updated = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0
            if self._refill <= 0.0:
                return False, 60
            return False, max(1, math.ceil((1.0 - bucket.tokens) / self._refill))

    def _evict_full(self) -> None:
        for key in [k for k, b in self._buckets.items() if b.tokens >= self._capacity]:
            del self._buckets[key]


def client_identity(scope: Scope) -> str:
    for name, value in scope.get("headers", []):
        if name == b"authorization":
            scheme, _, token = value.partition(b" ")
            if scheme.lower() == b"bearer" and token.strip():
                return "key:" + hashlib.sha256(token.strip()).hexdigest()[:16]
    client = scope.get("client")
    return f"ip:{client[0]}" if client else "ip:unknown"


class RateLimitMiddleware:
    """Shed a client exceeding its sustained request rate with 429 before the app does work."""

    def __init__(self, app: ASGIApp, *, limiter: RateLimiter | None = None) -> None:
        self._app = app
        self._limiter = limiter or RateLimiter()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in UNLIMITED_PATHS:
            await self._app(scope, receive, send)
            return
        allowed, retry_after = self._limiter.check(client_identity(scope))
        if not allowed:
            response = PlainTextResponse(
                "rate limit exceeded",
                status_code=int(HTTPStatus.TOO_MANY_REQUESTS),
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return
        await self._app(scope, receive, send)


class BodySizeLimitMiddleware:
    """Reject an oversized request body with 413 before the app buffers and parses it."""

    def __init__(self, app: ASGIApp, *, max_bytes: int = MAX_REQUEST_BYTES) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        if self._declared_length(scope) > self._max_bytes:
            response = PlainTextResponse("request body too large", status_code=413)
            await response(scope, receive, send)
            return
        await self._app(scope, self._guarded(receive), send)

    def _declared_length(self, scope: Scope) -> int:
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    return int(value)
                except ValueError:
                    return 0
        return 0

    def _guarded(self, receive: Receive) -> Receive:
        # Backstop a missing or untruthful Content-Length: once the cap is hit, stop reading and
        # keep reporting a disconnect so a streamed body cannot grow serving's memory without bound.
        received = 0
        over = False

        async def guarded() -> Message:
            nonlocal received, over
            if over:
                return {"type": "http.disconnect"}
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self._max_bytes:
                    over = True
                    return {"type": "http.disconnect"}
            return message

        return guarded
