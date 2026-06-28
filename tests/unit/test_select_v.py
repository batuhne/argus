import numpy as np
import pandas as pd

from fraud.features.select_v import (
    _drop_high_missing,
    _sample_rows,
    _v_columns,
    select_v_columns,
)
from fraud.params import SelectVParams
from fraud.transforms.features import LABEL_COLUMN


def _params(**overrides: object) -> SelectVParams:
    base = {
        "max_missing_fraction": 0.9,
        "correlation_threshold": 0.85,
        "max_features": 30,
        "corr_sample_rows": 100_000,
    }
    return SelectVParams(**{**base, **overrides})  # type: ignore[arg-type]


def test_v_columns_detects_only_numbered_v_fields() -> None:
    frame = pd.DataFrame(columns=["V1", "V12", "Vx", "V", "card1", LABEL_COLUMN])

    assert _v_columns(frame) == ["V1", "V12"]


def test_drop_high_missing_keeps_at_the_boundary() -> None:
    frame = pd.DataFrame({"V1": [1.0, 2.0, np.nan, np.nan], "V2": [1.0, np.nan, np.nan, np.nan]})

    # V1 is exactly 0.5 missing (kept), V2 is 0.75 (dropped).
    assert _drop_high_missing(frame, ["V1", "V2"], 0.5) == ["V1"]


def test_sample_rows_returns_frame_when_within_cap() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3]})

    assert _sample_rows(frame, max_rows=100, seed=0) is frame


def test_sample_rows_is_seeded_and_reproducible() -> None:
    frame = pd.DataFrame({"a": range(100)})

    first = _sample_rows(frame, max_rows=10, seed=42)
    second = _sample_rows(frame, max_rows=10, seed=42)

    assert len(first) == 10
    assert first.equals(second)


def _correlated_frame(rows: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, size=rows)
    signal = label + rng.normal(scale=0.1, size=rows)
    return pd.DataFrame(
        {
            "V1": signal,
            "V2": signal * 3.0 + 1.0,  # strictly monotonic in V1, so Spearman is 1.0
            "V3": rng.normal(size=rows),  # independent noise
            LABEL_COLUMN: label,
        }
    )


def test_clustering_keeps_one_representative_per_correlated_group() -> None:
    selected = select_v_columns(_correlated_frame(), _params(), seed=0)

    assert "V1" in selected  # stronger label association than its clustered twin
    assert "V2" not in selected  # dropped as a correlated follower
    assert "V3" in selected  # independent, its own cluster


def test_max_features_caps_the_selection() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({f"V{i}": rng.normal(size=2000) for i in range(1, 9)})
    frame[LABEL_COLUMN] = rng.integers(0, 2, size=2000)

    selected = select_v_columns(frame, _params(max_features=3), seed=0)

    assert len(selected) == 3


def test_selection_is_deterministic() -> None:
    frame = _correlated_frame()

    first = select_v_columns(frame, _params(), seed=0)
    second = select_v_columns(frame, _params(), seed=0)

    assert first == second
    assert first == sorted(first)


def test_returns_empty_when_no_v_columns() -> None:
    frame = pd.DataFrame({"card1": [1, 2], LABEL_COLUMN: [0, 1]})

    assert select_v_columns(frame, _params(), seed=0) == []
