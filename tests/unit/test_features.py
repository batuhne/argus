import numpy as np
import pandas as pd
import pytest

from fraud.transforms.features import FEATURE_COLUMNS, LABEL_COLUMN, build_xy


def _full_frame(rows: int = 4) -> pd.DataFrame:
    columns = {name: np.arange(rows, dtype="float32") for name in FEATURE_COLUMNS}
    columns[LABEL_COLUMN] = np.array([0, 1, 0, 1][:rows], dtype="int8")
    columns["TransactionID"] = np.arange(rows)
    return pd.DataFrame(columns)


def test_build_xy_returns_features_in_declared_order() -> None:
    x, _ = build_xy(_full_frame())

    assert tuple(x.columns) == FEATURE_COLUMNS


def test_build_xy_drops_extra_columns_from_x() -> None:
    x, _ = build_xy(_full_frame())

    assert "TransactionID" not in x.columns
    assert LABEL_COLUMN not in x.columns


def test_build_xy_label_is_int8() -> None:
    _, y = build_xy(_full_frame())

    assert y.dtype == np.int8


def test_build_xy_raises_when_features_missing() -> None:
    frame = _full_frame().drop(columns=["amt_log"])

    with pytest.raises(KeyError, match="amt_log"):
        build_xy(frame)


def test_build_xy_raises_when_label_missing() -> None:
    frame = _full_frame().drop(columns=[LABEL_COLUMN])

    with pytest.raises(KeyError, match=LABEL_COLUMN):
        build_xy(frame)
