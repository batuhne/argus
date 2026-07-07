from structlog.testing import capture_logs

from fraud.model_loader import ArtifactIntegrityError, FeatureContractError
from fraud.serving.reload import CHAMPION_RELOAD_FAILURES, ChampionReloader


def _reload_failures(reason: str) -> float:
    for sample in next(iter(CHAMPION_RELOAD_FAILURES.collect())).samples:
        if sample.name.endswith("_total") and sample.labels.get("reason") == reason:
            return float(sample.value)
    return 0.0


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
    before = _reload_failures("transient")

    with capture_logs() as logs:
        assert reloader.reload_once() is False

    assert _reload_failures("transient") == before + 1
    assert [e["log_level"] for e in logs if e["event"] == "champion_reload_failed"] == ["warning"]


def test_reload_once_rejects_a_bad_promotion_loudly() -> None:
    def bad() -> _Bundle:
        raise ArtifactIntegrityError("calibrator hash mismatch")

    reloader = _reloader(load=bad)
    before = _reload_failures("integrity")

    with capture_logs() as logs:
        assert reloader.reload_once() is False

    assert _reload_failures("integrity") == before + 1
    rejected = [e for e in logs if e["event"] == "champion_reload_rejected"]
    assert rejected and rejected[0]["log_level"] == "error"


def test_reload_once_flags_a_feature_contract_rejection() -> None:
    def bad() -> _Bundle:
        raise FeatureContractError("champion feature set does not match")

    reloader = _reloader(load=bad)
    before = _reload_failures("contract")

    assert reloader.reload_once() is False
    assert _reload_failures("contract") == before + 1


def test_start_does_not_spawn_a_thread_when_disabled() -> None:
    reloader = _reloader(interval_seconds=0.0)

    reloader.start()

    assert reloader._thread is None


def test_start_spawns_a_thread_and_stop_joins_it() -> None:
    reloader = _reloader(interval_seconds=0.05)

    reloader.start()
    assert reloader._thread is not None
    assert reloader._thread.is_alive()

    reloader.stop()
    assert not reloader._thread.is_alive()
