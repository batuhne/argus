"""Background hot-swap of the champion bundle when a newer version is promoted."""

from __future__ import annotations

import threading
from collections.abc import Callable

from fraud.common.logging import get_logger
from fraud.model_loader import ModelBundle

log = get_logger(__name__)

_STOP_JOIN_TIMEOUT_SECONDS = 5.0


class ChampionReloader:
    """Polls the champion alias and applies a freshly loaded bundle when the version moves."""

    def __init__(
        self,
        *,
        interval_seconds: float,
        current_version: Callable[[], int],
        latest_version: Callable[[], int | None],
        load: Callable[[], ModelBundle],
        apply: Callable[[ModelBundle], None],
    ) -> None:
        self._interval = interval_seconds
        self._current_version = current_version
        self._latest_version = latest_version
        self._load = load
        self._apply = apply
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._interval <= 0.0:
            return
        self._thread = threading.Thread(target=self._run, name="champion-reloader", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.reload_once()

    def reload_once(self) -> bool:
        try:
            latest = self._latest_version()
            if latest is None or latest == self._current_version():
                return False
            bundle = self._load()
            self._apply(bundle)
            log.info("champion_reloaded", version=bundle.version, family=bundle.family)
            return True
        except Exception as exc:
            log.warning("champion_reload_failed", error=str(exc), exc_info=True)
            return False

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=_STOP_JOIN_TIMEOUT_SECONDS)
