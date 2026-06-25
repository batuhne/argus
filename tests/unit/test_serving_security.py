import asyncio
import types
from typing import Any

import pytest
from starlette.datastructures import Headers
from starlette.types import Message, Receive, Scope, Send

from fraud.serving.security import (
    BodySizeLimitMiddleware,
    Unauthorized,
    verify_api_key,
)
from fraud.serving.service import FraudService


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
