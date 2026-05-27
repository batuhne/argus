"""Assemble a point-in-time correct training frame from the feature store.

For each transaction Feast joins the feature values as of that timestamp, so the
frame never holds anything from the future.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud.common.logging import configure_logging, get_logger
from fraud.config import get_settings
from fraud.features.store import open_feature_store
from fraud.paths import FEATURE_REPO_DIR, PROCESSED_DIR
from fraud.transforms import feature_logic as fl

FEATURE_SERVICE = "card_activity"
LABEL_COLUMN = "isFraud"
SOURCE_COLUMNS = [
    "TransactionID",
    *fl.IDENTITY_COLUMNS,
    "TransactionDT",
    "TransactionAmt",
    LABEL_COLUMN,
]

log = get_logger(__name__)


def _entity_frame(split_path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(split_path, columns=SOURCE_COLUMNS)
    keyed = fl.add_keys_and_timestamp(raw)
    return keyed[["TransactionID", "card_id", "event_timestamp", "TransactionAmt", LABEL_COLUMN]]


def build_training_frame(
    split: str = "train",
    repo_dir: Path = FEATURE_REPO_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    entity_df = _entity_frame(processed_dir / f"{split}.parquet")
    store = open_feature_store(repo_dir)
    job = store.get_historical_features(
        entity_df=entity_df,
        features=store.get_feature_service(FEATURE_SERVICE),
    )
    return job.to_df()


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    frame = build_training_frame("train")
    log.info(
        "training_frame_built",
        rows=len(frame),
        columns=frame.shape[1],
        fraud_rate=round(float(frame[LABEL_COLUMN].mean()), 6),
    )


if __name__ == "__main__":
    main()
