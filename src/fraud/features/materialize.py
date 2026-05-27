"""Register the feature views and load the latest values into Redis.

Run after build_offline, with the Redis service up.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from fraud.features.registry import default_objects
from fraud.features.store import open_feature_store
from fraud.paths import CARD_FEATURES_PATH, FEATURE_REPO_DIR


def _materialization_window(features_path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    timestamps = pd.read_parquet(features_path, columns=["event_timestamp"])["event_timestamp"]
    # Nudge past the last event so it is included.
    return timestamps.min(), timestamps.max() + timedelta(seconds=1)


def materialize(
    repo_dir: Path = FEATURE_REPO_DIR, features_path: Path = CARD_FEATURES_PATH
) -> None:
    store = open_feature_store(repo_dir)
    store.apply(default_objects().to_list())

    start, end = _materialization_window(features_path)
    store.materialize(start_date=start.to_pydatetime(), end_date=end.to_pydatetime())


if __name__ == "__main__":
    materialize()
