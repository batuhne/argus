"""Feature transforms shared by training and serving.

Defining every feature once, here, is what keeps offline and online values
identical. The offline builder, the Feast views, and the stream all import this.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# TransactionDT is a seconds offset; anchoring it to a fixed instant makes the
# timestamps deterministic. The absolute date does not matter.
REFERENCE_DATETIME = datetime(2017, 12, 1, tzinfo=UTC)

# The numeric issuer fields that identify a card. card4 (brand) and card6 (type)
# describe it but do not identify it.
IDENTITY_COLUMNS = ("card1", "card2", "card3", "card5")

VELOCITY_WINDOW = "24h"
VELOCITY_COLUMNS = (
    "card_txn_count_24h",
    "card_amt_sum_24h",
    "card_amt_mean_24h",
    "card_amt_max_24h",
    "seconds_since_prev_txn",
)

# seconds_since_prev_txn when the card has no earlier transaction.
NO_PRIOR_TXN = -1.0

# Per-transaction C (counts), D (time deltas), distance and address fields. They
# ride the request and reach the trees with NaN intact: missingness is signal.
C_COLUMNS = tuple(f"C{i}" for i in range(1, 15))
D_COLUMNS = tuple(f"D{i}" for i in range(1, 16))
DIST_COLUMNS = ("dist1", "dist2")
ADDR_COLUMNS = ("addr1", "addr2")
RAW_NUMERIC_PASSTHROUGH = (*C_COLUMNS, *D_COLUMNS, *DIST_COLUMNS, *ADDR_COLUMNS)

# Per-transaction categoricals the encoder turns into frequency and smoothed-target
# features. Curated by hand, unlike the offline-frozen V set; the fitted maps are
# persisted with the model, so only the names live here.
CATEGORICAL_COLUMNS = (
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "DeviceType",
    "DeviceInfo",
    "id_12",
    "id_15",
    "id_16",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
)

# The reduced V set is frozen offline by select_v; reading it here means the same
# list feeds training and serving. Anchored to the repo root (not the process CWD)
# so both sides resolve the same file, and empty on a fresh checkout so imports never fail.
V_SELECTED_PATH = Path(__file__).resolve().parents[3] / "feature_repo" / "v_selected.json"


def load_v_selected(path: Path = V_SELECTED_PATH) -> tuple[str, ...]:
    if not path.exists():
        return ()
    return tuple(json.loads(path.read_text()))


V_SELECTED = load_v_selected()


def to_event_timestamp(transaction_dt: pd.Series) -> pd.Series:
    return REFERENCE_DATETIME + pd.to_timedelta(transaction_dt.astype("int64"), unit="s")


def make_card_id(frame: pd.DataFrame) -> pd.Series:
    # Missing parts collapse to a sentinel so the key is always defined.
    parts = [frame[col].astype("Int64").astype("string").fillna("na") for col in IDENTITY_COLUMNS]
    card_id = parts[0]
    for part in parts[1:]:
        card_id = card_id.str.cat(part, sep="_")
    return card_id.astype(str)


def add_keys_and_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["card_id"] = make_card_id(frame)
    out["event_timestamp"] = to_event_timestamp(frame["TransactionDT"])
    return out


def compute_card_velocity(frame: pd.DataFrame) -> pd.DataFrame:
    # Left-closed window: each transaction sees only earlier ones on the same
    # card, which is what serving knows before the event is recorded. Empty
    # windows (a first or long-dormant card) fill to zero.
    work = (
        frame[["card_id", "event_timestamp", "TransactionAmt"]]
        .sort_values(["card_id", "event_timestamp"], kind="mergesort")
        .reset_index(drop=True)
    )
    rolling = (
        work.set_index("event_timestamp")
        .groupby("card_id", sort=True)["TransactionAmt"]
        .rolling(VELOCITY_WINDOW, closed="left")
    )
    gap = work.groupby("card_id", sort=True)["event_timestamp"].diff().dt.total_seconds()

    out = work[["card_id", "event_timestamp"]].copy()
    out["card_txn_count_24h"] = rolling.count().fillna(0.0).to_numpy().astype("float32")
    out["card_amt_sum_24h"] = rolling.sum().fillna(0.0).to_numpy().astype("float32")
    out["card_amt_mean_24h"] = rolling.mean().fillna(0.0).to_numpy().astype("float32")
    out["card_amt_max_24h"] = rolling.max().fillna(0.0).to_numpy().astype("float32")
    out["seconds_since_prev_txn"] = gap.fillna(NO_PRIOR_TXN).to_numpy().astype("float32")
    return out


def coerce_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Cast passthrough columns to float32; non-finite (incl. float32 overflow) becomes NaN."""
    out = frame.copy()
    for column in columns:
        coerced = pd.to_numeric(frame[column], errors="coerce").astype("float32")
        out[column] = coerced.where(np.isfinite(coerced))
    return out


def amount_log(amount: pd.Series) -> pd.Series:
    return pd.Series(np.log1p(amount.to_numpy()), index=amount.index, name="amt_log")


def amount_to_mean_ratio(amount: pd.Series, mean_amount: pd.Series) -> pd.Series:
    # No prior history gives a neutral ratio of 1: a zero mean, a missing value,
    # or a never-seen card whose online lookup returns null. Coercing first keeps
    # that null (Python None on an object column) from raising at serving time.
    amt = pd.to_numeric(amount, errors="coerce").to_numpy(dtype="float64")
    safe_mean = (
        pd.to_numeric(mean_amount, errors="coerce").replace(0.0, np.nan).to_numpy(dtype="float64")
    )
    ratio = amt / safe_mean
    return pd.Series(ratio, index=amount.index, name="amt_to_card_mean_24h").fillna(1.0)
