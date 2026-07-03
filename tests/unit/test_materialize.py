from pathlib import Path

import pandas as pd
import pytest

from fraud.features.materialize import _materialization_window


def test_materialization_window_raises_on_empty_features(tmp_path: Path) -> None:
    path = tmp_path / "card_features.parquet"
    pd.DataFrame({"event_timestamp": pd.Series([], dtype="datetime64[ns]")}).to_parquet(path)

    with pytest.raises(ValueError, match="event_timestamp"):
        _materialization_window(path)
