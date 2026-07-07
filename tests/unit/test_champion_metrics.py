import types
from typing import Any

import pytest

from fraud.evaluation.gate import GateMetrics
from fraud.registry import CHAMPION_TAG_AUPRC, CHAMPION_TAG_COST_PER_TX, ModelFamily
from fraud.training import train


def _cfg() -> Any:
    return types.SimpleNamespace(model_name="argus", champion_alias="champion")


def test_load_champion_metrics_is_none_without_a_champion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "get_alias_version", lambda _n, _a: None)

    assert train._load_champion_metrics(_cfg()) is None


def test_load_champion_metrics_reads_present_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "get_alias_version", lambda _n, _a: 1)
    monkeypatch.setattr(
        train,
        "get_version_tags",
        lambda _n, _v: {CHAMPION_TAG_AUPRC: "0.8", CHAMPION_TAG_COST_PER_TX: "1.5"},
    )

    assert train._load_champion_metrics(_cfg()) == GateMetrics(auprc=0.8, expected_cost_per_tx=1.5)


def test_load_champion_metrics_raises_on_missing_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "get_alias_version", lambda _n, _a: 1)
    monkeypatch.setattr(train, "get_version_tags", lambda _n, _v: {})

    with pytest.raises(RuntimeError, match="missing gate tags"):
        train._load_champion_metrics(_cfg())


def test_load_champion_metrics_raises_on_non_finite_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "get_alias_version", lambda _n, _a: 1)
    monkeypatch.setattr(
        train,
        "get_version_tags",
        lambda _n, _v: {CHAMPION_TAG_AUPRC: "nan", CHAMPION_TAG_COST_PER_TX: "1.5"},
    )

    with pytest.raises(RuntimeError, match="non-finite gate tags"):
        train._load_champion_metrics(_cfg())


def test_promote_writes_tags_before_moving_the_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    # If the tag write fails, the alias must not move, so the old champion stays servable.
    aliased: list[int] = []

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("registry write failed")

    monkeypatch.setattr(train, "write_version_tags", boom)
    monkeypatch.setattr(train, "attach_alias", lambda *_a, **_k: aliased.append(1))

    with pytest.raises(RuntimeError, match="registry write failed"):
        train._promote_champion(
            5,
            GateMetrics(auprc=0.8, expected_cost_per_tx=1.0),
            _cfg(),
            threshold=0.5,
            family=ModelFamily.XGBOOST,
        )

    assert aliased == []
