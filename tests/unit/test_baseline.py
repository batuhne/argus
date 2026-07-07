import dataclasses
import time
from typing import Any

import pandas as pd
import pytest

from fraud.monitoring.baseline import BaselineUnavailableError, load_baseline
from fraud.monitoring.config import MonitoringConfig


def _cfg(**overrides: Any) -> MonitoringConfig:
    return dataclasses.replace(MonitoringConfig.from_settings(), **overrides)


def test_load_baseline_retries_until_the_registry_is_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({"a": [1.0]})
    calls = {"n": 0}

    def fake_load(_cfg: MonitoringConfig) -> pd.DataFrame:
        calls["n"] += 1
        if calls["n"] < 3:
            raise BaselineUnavailableError("not yet")
        return frame

    monkeypatch.setattr("fraud.monitoring.baseline._load_baseline", fake_load)
    monkeypatch.setattr("mlflow.set_tracking_uri", lambda _uri: None)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    result = load_baseline(_cfg(baseline_load_deadline_seconds=30.0))

    assert result is frame
    assert calls["n"] == 3


def test_load_baseline_gives_up_after_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_unavailable(_cfg: MonitoringConfig) -> pd.DataFrame:
        raise BaselineUnavailableError("never ready")

    monkeypatch.setattr("fraud.monitoring.baseline._load_baseline", always_unavailable)
    monkeypatch.setattr("mlflow.set_tracking_uri", lambda _uri: None)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with pytest.raises(BaselineUnavailableError):
        load_baseline(_cfg(baseline_load_deadline_seconds=0.0))
