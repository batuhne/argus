from fraud.serving.reload import ChampionReloader


class _Bundle:
    def __init__(self, version: int) -> None:
        self.version = version
        self.family = "xgboost"


def _reloader(**overrides: object) -> ChampionReloader:
    loads = {"count": 0}

    def load() -> _Bundle:
        loads["count"] += 1
        return _Bundle(2)

    kwargs: dict[str, object] = {
        "interval_seconds": 0.0,
        "current_version": lambda: 1,
        "latest_version": lambda: 2,
        "load": load,
        "apply": lambda bundle: None,
    }
    kwargs.update(overrides)
    reloader = ChampionReloader(**kwargs)  # type: ignore[arg-type]
    reloader._loads = loads  # type: ignore[attr-defined]
    return reloader


def test_reload_once_applies_when_the_version_advances() -> None:
    applied: list[int] = []
    reloader = _reloader(apply=lambda bundle: applied.append(bundle.version))

    assert reloader.reload_once() is True
    assert applied == [2]


def test_reload_once_is_noop_when_the_version_is_unchanged() -> None:
    reloader = _reloader(latest_version=lambda: 1)

    assert reloader.reload_once() is False
    assert reloader._loads["count"] == 0  # type: ignore[attr-defined]


def test_reload_once_is_noop_when_the_alias_is_missing() -> None:
    reloader = _reloader(latest_version=lambda: None)

    assert reloader.reload_once() is False
    assert reloader._loads["count"] == 0  # type: ignore[attr-defined]


def test_reload_once_survives_a_loader_failure() -> None:
    def boom() -> _Bundle:
        raise RuntimeError("mlflow unreachable")

    reloader = _reloader(load=boom)

    assert reloader.reload_once() is False


def test_start_does_not_spawn_a_thread_when_disabled() -> None:
    reloader = _reloader(interval_seconds=0.0)

    reloader.start()

    assert reloader._thread is None
