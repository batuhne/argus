"""Build the offline feature table that backs the Feast file source."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud.paths import CARD_FEATURES_PATH, INTERIM_DIR
from fraud.transforms import feature_logic as fl

INPUT_PATH = INTERIM_DIR / "clean.parquet"
SOURCE_COLUMNS = [*fl.IDENTITY_COLUMNS, "TransactionDT", "TransactionAmt"]


def build_offline_features(
    input_path: Path = INPUT_PATH, output_path: Path = CARD_FEATURES_PATH
) -> None:
    raw = pd.read_parquet(input_path, columns=SOURCE_COLUMNS)
    keyed = fl.add_keys_and_timestamp(raw)
    features = fl.compute_card_velocity(keyed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    features.to_parquet(tmp, index=False)
    tmp.replace(output_path)


if __name__ == "__main__":
    build_offline_features()
