import asyncio
import types
from typing import Any

import pytest
from pydantic import SecretStr
from starlette.datastructures import Headers
from starlette.types import Message, Receive, Scope, Send

from fraud.serving.security import (
    BodySizeLimitMiddleware,
    RateLimiter,
    RateLimitMiddleware,
    Unauthorized,
    client_identity,
    verify_api_key,
)
from fraud.serving.service import FraudService, _resolve_api_key


@pytest.mark.parametrize("authorization", ["Bearer secret", "Bearer  secret  ", "bearer secret"])
def test_verify_api_key_accepts_matching_bearer_token(authorization: str) -> None:
    verify_api_key(authorization, "secret")


@pytest.mark.parametrize(
    "authorization", [None, "", "secret", "Bearer wrong", "Bearer ", "Basic secret"]
)
def test_verify_api_key_rejects_bad_credentials(authorization: str | None) -> None:
    with pytest.raises(Unauthorized):
        verify_api_key(authorization, "secret")


def _service_with_key(api_key: str | None) -> Any:
    inner = FraudService.inner  # type: ignore[attr-defined]
    service = inner.__new__(inner)  # skip __init__ and its model load
    service._api_key = api_key
    return service


def _ctx(headers: dict[str, str]) -> types.SimpleNamespace:
    return types.SimpleNamespace(request=types.SimpleNamespace(headers=Headers(headers)))


def test_authenticate_passes_with_a_valid_key() -> None:
    _service_with_key("secret")._authenticate(_ctx({"Authorization": "Bearer secret"}))


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}])
def test_authenticate_rejects_missing_or_wrong_key(headers: dict[str, str]) -> None:
    with pytest.raises(Unauthorized):
        _service_with_key("secret")._authenticate(_ctx(headers))


def test_authenticate_is_a_noop_when_auth_is_disabled() -> None:
    _service_with_key(None)._authenticate(_ctx({}))


class _SpyApp:
    def __init__(self) -> None:
        self.called = False
        self.body = b""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.called = True
        more = True
        while more:
            message = await receive()
            if message["type"] == "http.request":
                self.body += message.get("body", b"")
                more = message.get("more_body", False)
            else:
                more = False
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


async def _drive(
    middleware: BodySizeLimitMiddleware, *, content_length: int | None, chunks: list[bytes]
) -> int:
    headers = [] if content_length is None else [(b"content-length", str(content_length).encode())]
    pending = list(chunks)

    async def receive() -> Message:
        if pending:
            body = pending.pop(0)
            return {"type": "http.request", "body": body, "more_body": bool(pending)}
        return {"type": "http.disconnect"}

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    await middleware({"type": "http", "headers": headers}, receive, send)
    return int(next(m["status"] for m in sent if m["type"] == "http.response.start"))


def test_body_under_limit_reaches_the_app() -> None:
    app = _SpyApp()
    middleware = BodySizeLimitMiddleware(app, max_bytes=16)

    status = asyncio.run(_drive(middleware, content_length=8, chunks=[b"x" * 8]))

    assert status == 200
    assert app.called
    assert app.body == b"x" * 8


def test_declared_oversize_body_is_rejected_before_the_app() -> None:
    app = _SpyApp()
    middleware = BodySizeLimitMiddleware(app, max_bytes=16)

    status = asyncio.run(_drive(middleware, content_length=100, chunks=[b"x" * 100]))

    assert status == 413
    assert not app.called


def test_streamed_body_is_cut_off_at_the_limit_without_a_declared_length() -> None:
    app = _SpyApp()
    middleware = BodySizeLimitMiddleware(app, max_bytes=16)

    status = asyncio.run(_drive(middleware, content_length=None, chunks=[b"x" * 10, b"x" * 10]))

    assert status == 200  # the app runs, but never sees more than the cap
    assert len(app.body) <= 16


def test_non_http_scope_passes_through_untouched() -> None:
    app = _SpyApp()
    middleware = BodySizeLimitMiddleware(app, max_bytes=16)

    async def run() -> None:
        async def receive() -> Message:
            return {"type": "lifespan.startup"}

        async def send(message: Message) -> None:
            return None

        await middleware({"type": "lifespan"}, receive, send)

    asyncio.run(run())
    assert app.called


def test_resolve_api_key_returns_secret_value_when_set() -> None:
    assert _resolve_api_key(SecretStr("k"), "production") == "k"


def test_resolve_api_key_allows_none_in_local() -> None:
    assert _resolve_api_key(None, "local") is None


@pytest.mark.parametrize("environment", ["production", "staging"])
def test_resolve_api_key_refuses_none_outside_local(environment: str) -> None:
    with pytest.raises(RuntimeError, match="serving_api_key"):
        _resolve_api_key(None, environment)


def test_rate_limiter_allows_up_to_capacity_then_blocks() -> None:
    limiter = RateLimiter(capacity=3, refill_per_second=1.0, clock=lambda: 0.0)

    assert [limiter.check("k")[0] for _ in range(3)] == [True, True, True]
    allowed, retry_after = limiter.check("k")

    assert allowed is False
    assert retry_after >= 1


def test_rate_limiter_refills_over_time() -> None:
    now = {"t": 0.0}
    limiter = RateLimiter(capacity=1, refill_per_second=2.0, clock=lambda: now["t"])

    assert limiter.check("k")[0] is True
    assert limiter.check("k")[0] is False
    now["t"] = 0.5

    assert limiter.check("k")[0] is True


def test_rate_limiter_isolates_clients() -> None:
    limiter = RateLimiter(capacity=1, refill_per_second=1.0, clock=lambda: 0.0)

    assert limiter.check("a")[0] is True
    assert limiter.check("b")[0] is True
    assert limiter.check("a")[0] is False


def test_rate_limiter_caps_the_identity_table_under_a_flood() -> None:
    # Every client spends a token so no bucket is idle; the table must still stay bounded,
    # evicting the least-recently-used identities rather than growing without end.
    limiter = RateLimiter(capacity=2, refill_per_second=0.0, max_clients=3, clock=lambda: 0.0)

    for i in range(50):
        limiter.check(f"client-{i}")

    assert len(limiter._buckets) <= 3
    assert "client-49" in limiter._buckets
    assert "client-0" not in limiter._buckets


def test_client_identity_is_the_source_ip() -> None:
    scope: Scope = {"type": "http", "headers": [], "client": ("1.2.3.4", 5)}

    assert client_identity(scope) == "ip:1.2.3.4"


def test_client_identity_is_unknown_without_a_client() -> None:
    assert client_identity({"type": "http", "headers": []}) == "ip:unknown"


async def _drive_rate_limit(
    middleware: RateLimitMiddleware,
    path: str = "/predict",
    *,
    ip: str = "1.2.3.4",
    authorization: str | None = None,
) -> int:
    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    headers = [(b"authorization", authorization.encode())] if authorization is not None else []
    scope: Scope = {"type": "http", "path": path, "headers": headers, "client": (ip, 1)}
    await middleware(scope, receive, send)
    return int(next(m["status"] for m in sent if m["type"] == "http.response.start"))


def test_rate_limit_middleware_sheds_a_client_over_its_limit() -> None:
    app = _SpyApp()
    limiter = RateLimiter(capacity=1, refill_per_second=0.001, clock=lambda: 0.0)
    middleware = RateLimitMiddleware(app, limiter=limiter)

    first = asyncio.run(_drive_rate_limit(middleware))
    second = asyncio.run(_drive_rate_limit(middleware))

    assert first == 200
    assert second == 429


def test_rate_limit_middleware_exempts_the_verified_internal_key() -> None:
    app = _SpyApp()
    limiter = RateLimiter(capacity=1, refill_per_second=0.001, clock=lambda: 0.0)
    middleware = RateLimitMiddleware(app, limiter=limiter, expected_key="secret")

    statuses = [
        asyncio.run(_drive_rate_limit(middleware, authorization="Bearer secret")) for _ in range(5)
    ]

    assert statuses == [200, 200, 200, 200, 200]


def test_rate_limit_middleware_buckets_a_rotating_token_flood_by_ip() -> None:
    app = _SpyApp()
    limiter = RateLimiter(capacity=1, refill_per_second=0.001, clock=lambda: 0.0)
    middleware = RateLimitMiddleware(app, limiter=limiter, expected_key="secret")

    first = asyncio.run(_drive_rate_limit(middleware, authorization="Bearer forged-1"))
    second = asyncio.run(_drive_rate_limit(middleware, authorization="Bearer forged-2"))

    assert first == 200
    assert second == 429  # a fresh token per request no longer buys a fresh bucket


@pytest.mark.parametrize("path", ["/livez", "/readyz", "/metrics"])
def test_rate_limit_middleware_never_throttles_probes(path: str) -> None:
    app = _SpyApp()
    limiter = RateLimiter(capacity=1, refill_per_second=0.001, clock=lambda: 0.0)
    middleware = RateLimitMiddleware(app, limiter=limiter)

    statuses = [asyncio.run(_drive_rate_limit(middleware, path)) for _ in range(3)]

    assert statuses == [200, 200, 200]
