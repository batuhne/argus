"""Request-time controls for the serving API: bearer auth, a body-size guard, and rate limiting."""

from __future__ import annotations

import hmac
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus

from bentoml.exceptions import BentoMLException
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Comfortably above a legit predict body (scalars plus the bounded v and categorical maps).
MAX_REQUEST_BYTES = 64 * 1024

# Caps one source IP's sustained rate; max_concurrency only bounds global in-flight, not per-IP.
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
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, identity: str) -> tuple[bool, int]:
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(identity)
            if bucket is None:
                self._make_room()
                bucket = _Bucket(self._capacity, now)
                self._buckets[identity] = bucket
            else:
                self._buckets.move_to_end(identity)
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

    def _make_room(self) -> None:
        # A distinct-identity flood keeps every bucket drained, so evict least-recently-used
        # until there is room; a full table can never grow past max_clients.
        while len(self._buckets) >= self._max_clients:
            self._buckets.popitem(last=False)


def client_identity(scope: Scope) -> str:
    client = scope.get("client")
    return f"ip:{client[0]}" if client else "ip:unknown"


def _bearer_token(scope: Scope) -> bytes:
    for name, value in scope.get("headers", []):
        if name == b"authorization":
            scheme, _, token = value.partition(b" ")
            if scheme.lower() == b"bearer":
                return bytes(token.strip())
    return b""


class RateLimitMiddleware:
    """Shed untrusted traffic over its per-IP rate with 429; the verified internal key is exempt."""

    def __init__(
        self, app: ASGIApp, *, limiter: RateLimiter | None = None, expected_key: str | None = None
    ) -> None:
        self._app = app
        self._limiter = limiter or RateLimiter()
        self._expected_key = expected_key.encode("utf-8") if expected_key else None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in UNLIMITED_PATHS:
            await self._app(scope, receive, send)
            return
        # The internal key skips shedding; a leaked key is unthrottled, so rotate it on exposure.
        if self._is_trusted(scope):
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

    def _is_trusted(self, scope: Scope) -> bool:
        if self._expected_key is None:
            return False
        token = _bearer_token(scope)
        return bool(token) and hmac.compare_digest(token, self._expected_key)


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
