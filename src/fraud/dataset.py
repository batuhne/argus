"""Point-in-time feature joins (merge_asof) and out-of-fold categorical encoding for the splits."""

from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd

from fraud.common.logging import configure_logging, get_logger
from fraud.config import get_settings
from fraud.paths import FEATURE_REPO_DIR, PROCESSED_DIR
from fraud.transforms import feature_logic as fl
from fraud.transforms.encoders import CategoricalEncoder, fit_transform_oof
from fraud.transforms.features import LABEL_COLUMN

SPLITS: tuple[str, ...] = ("train", "val", "test")
SOURCE_COLUMNS = [
    "TransactionID",
    *fl.IDENTITY_COLUMNS,
    "TransactionDT",
    "TransactionAmt",
    *fl.RAW_NUMERIC_PASSTHROUGH,
    *fl.V_SELECTED,
    *fl.CATEGORICAL_COLUMNS,
    LABEL_COLUMN,
]
ENTITY_COLUMNS = [
    "TransactionID",
    "card_id",
    "event_timestamp",
    "TransactionAmt",
    LABEL_COLUMN,
    *fl.RAW_NUMERIC_PASSTHROUGH,
    *fl.V_SELECTED,
    *fl.CATEGORICAL_COLUMNS,
]

log = get_logger(__name__)


def _entity_frame(split_path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(split_path, columns=SOURCE_COLUMNS)
    keyed = fl.add_keys_and_timestamp(raw)
    return keyed[ENTITY_COLUMNS]


def _load_features(repo_dir: Path) -> pd.DataFrame:
    features = pd.read_parquet(repo_dir / "data" / "card_features.parquet")
    return features.sort_values("event_timestamp").reset_index(drop=True)


def _join_split(entity_df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    entity_sorted = entity_df.sort_values("event_timestamp").reset_index(drop=True)
    merged = pd.merge_asof(
        entity_sorted,
        features,
        on="event_timestamp",
        by="card_id",
        direction="backward",
    )
    merged["amt_log"] = fl.amount_log(merged["TransactionAmt"])
    merged["amt_to_card_mean_24h"] = fl.amount_to_mean_ratio(
        merged["TransactionAmt"], merged["card_amt_mean_24h"]
    )
    return fl.coerce_numeric(merged, [*fl.RAW_NUMERIC_PASSTHROUGH, *fl.V_SELECTED])


def build_training_frame(
    split: str = "train",
    repo_dir: Path = FEATURE_REPO_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    features = _load_features(repo_dir)
    entity_df = _entity_frame(processed_dir / f"{split}.parquet")
    return _join_split(entity_df, features)


def load_splits(
    repo_dir: Path = FEATURE_REPO_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> dict[str, pd.DataFrame]:
    features = _load_features(repo_dir)
    result: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        log.info("loading_split", split=split)
        entity_df = _entity_frame(processed_dir / f"{split}.parquet")
        frame = _join_split(entity_df, features)
        log.info(
            "split_loaded",
            split=split,
            rows=len(frame),
            mem_mb=int(frame.memory_usage(deep=True).sum() / 1e6),
        )
        result[split] = frame
        del entity_df
        gc.collect()
    del features
    gc.collect()
    return result


def build_eval_frame(
    split: str,
    encoder: CategoricalEncoder,
    repo_dir: Path = FEATURE_REPO_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """Build a labeled split through the shared join path and encode it with a fitted encoder.

    Transform only, no refit, so no label from the evaluated rows leaks into its own features.
    """
    frame = build_training_frame(split, repo_dir, processed_dir)
    _attach_encoded(frame, encoder.transform(frame))
    return frame


def add_encoded_categoricals(
    frames: dict[str, pd.DataFrame], *, seed: int, smoothing: float, n_splits: int
) -> CategoricalEncoder:
    """Fit the encoder on train and add encoded columns to every split in place.

    Train rows get leak-free out-of-fold target values; val and test use the full-train
    maps. The returned encoder holds those full-train maps, so serving encodes identically.
    """
    encoder, train_encoded = fit_transform_oof(
        frames["train"],
        fl.CATEGORICAL_COLUMNS,
        LABEL_COLUMN,
        seed=seed,
        smoothing=smoothing,
        n_splits=n_splits,
    )
    _attach_encoded(frames["train"], train_encoded)
    for split, frame in frames.items():
        if split != "train":
            _attach_encoded(frame, encoder.transform(frame))
    return encoder


def _attach_encoded(frame: pd.DataFrame, encoded: pd.DataFrame) -> None:
    for column in encoded.columns:
        frame[column] = encoded[column]


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
