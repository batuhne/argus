from __future__ import annotations

import pandas as pd

from fraud.transforms import feature_logic as fl

FEATURE_COLUMNS: tuple[str, ...] = (
    "card_txn_count_24h",
    "card_amt_sum_24h",
    "card_amt_mean_24h",
    "card_amt_max_24h",
    "seconds_since_prev_txn",
    "amt_to_card_mean_24h",
    "amt_log",
    "TransactionAmt",
    *fl.RAW_NUMERIC_PASSTHROUGH,
    *fl.V_SELECTED,
)
LABEL_COLUMN = "isFraud"


def build_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing_features = [name for name in FEATURE_COLUMNS if name not in frame.columns]
    if missing_features:
        raise KeyError(f"frame is missing feature columns: {missing_features}")
    if LABEL_COLUMN not in frame.columns:
        raise KeyError(f"frame is missing label column: {LABEL_COLUMN}")
    x = frame.loc[:, list(FEATURE_COLUMNS)]
    y = frame[LABEL_COLUMN].astype("int8")
    return x, y
