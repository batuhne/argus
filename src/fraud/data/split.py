from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pandas as pd

from fraud.params import Params, load_params
from fraud.paths import INTERIM_DIR, PROCESSED_DIR

INPUT_PATH = INTERIM_DIR / "clean.parquet"
TIME_COLUMN = "TransactionDT"


def time_split(
    df: pd.DataFrame, val_fraction: float, test_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically by transaction time.

    Train is the earliest window, test the latest. A random split would leak
    future information and ignore that fraud labels arrive late.
    """
    ordered = df.sort_values(TIME_COLUMN).reset_index(drop=True)
    total = len(ordered)
    test_n = int(total * test_fraction)
    val_n = int(total * val_fraction)
    train_n = total - val_n - test_n
    train = ordered.iloc[:train_n]
    val = ordered.iloc[train_n : train_n + val_n]
    test = ordered.iloc[train_n + val_n :]
    _assert_chronological(train, val, test)
    return train, val, test


def _assert_chronological(*splits: pd.DataFrame) -> None:
    bounds = [(s[TIME_COLUMN].min(), s[TIME_COLUMN].max()) for s in splits if len(s)]
    for (_, prev_max), (next_min, _) in pairwise(bounds):
        if prev_max >= next_min:
            raise ValueError(
                f"time split overlap: a split ends at {prev_max} "
                f"while the next starts at {next_min}"
            )


def split(
    input_path: Path = INPUT_PATH,
    output_dir: Path = PROCESSED_DIR,
    params: Params | None = None,
) -> None:
    settings = params or load_params()
    df = pd.read_parquet(input_path)
    train, val, test = time_split(df, settings.split.val_fraction, settings.split.test_fraction)

    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(output_dir / "train.parquet", index=False)
    val.to_parquet(output_dir / "val.parquet", index=False)
    test.to_parquet(output_dir / "test.parquet", index=False)


if __name__ == "__main__":
    split()
