"""Request-time controls for the serving API: bearer auth and a body-size guard."""

from __future__ import annotations

import hmac
from http import HTTPStatus

from bentoml.exceptions import BentoMLException
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Comfortably above a legit predict body (scalars plus the bounded v and categorical maps).
MAX_REQUEST_BYTES = 64 * 1024


class Unauthorized(BentoMLException):
    error_code = HTTPStatus.UNAUTHORIZED


def verify_api_key(authorization: str | None, expected: str) -> None:
    """Raise Unauthorized unless the request carries the expected bearer token."""
    scheme, _, token = (authorization or "").partition(" ")
    provided = token.strip() if scheme.lower() == "bearer" else ""
    # Constant-time compare so a wrong key cannot be guessed byte by byte from response timing.
    if not provided or not hmac.compare_digest(provided, expected):
        raise Unauthorized("missing or invalid API key")


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
