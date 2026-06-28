from __future__ import annotations

import signal
from types import FrameType


class ShutdownFlag:
    """Set on SIGINT/SIGTERM so a poll loop can exit cleanly between iterations."""

    def __init__(self) -> None:
        self.requested = False

    def request(self, _signum: int, _frame: FrameType | None) -> None:
        self.requested = True


def install_shutdown_handler() -> ShutdownFlag:
    shutdown = ShutdownFlag()
    signal.signal(signal.SIGINT, shutdown.request)
    signal.signal(signal.SIGTERM, shutdown.request)
    return shutdown
