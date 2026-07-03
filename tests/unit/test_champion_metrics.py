import types
from typing import Any

import pytest

from fraud.evaluation.gate import GateMetrics
from fraud.registry import CHAMPION_TAG_AUPRC, CHAMPION_TAG_COST_PER_TX
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
