"""Freeze a reduced V-feature set offline: drop sparse columns, de-correlate, keep the strongest."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from fraud.common.logging import configure_logging, get_logger
from fraud.config import get_settings
from fraud.params import SelectVParams, load_params
from fraud.paths import PROCESSED_DIR
from fraud.transforms.feature_logic import V_SELECTED_PATH
from fraud.transforms.features import LABEL_COLUMN

V_COLUMN_PREFIX = "V"
# A Spearman value between two card features needs enough overlapping non-null rows
# to mean anything; below this the pair stays uncorrelated rather than clustering on noise.
MIN_CORR_OBSERVATIONS = 1000

log = get_logger(__name__)


def select_v_columns(train: pd.DataFrame, params: SelectVParams, seed: int) -> list[str]:
    """Return the V columns to keep: dense, mutually de-correlated, most label-associated."""
    candidates = _v_columns(train)
    dense = _drop_high_missing(train, candidates, params.max_missing_fraction)
    if not dense:
        return []
    sample = _sample_rows(train[[*dense, LABEL_COLUMN]], params.corr_sample_rows, seed)
    corr = sample.corr(method="spearman", min_periods=MIN_CORR_OBSERVATIONS)
    strength = _label_association(corr, dense)
    representatives = _decorrelate(
        corr.loc[dense, dense].abs(), strength, params.correlation_threshold
    )
    strongest = sorted(representatives, key=lambda col: strength[col], reverse=True)
    return sorted(strongest[: params.max_features])


def _sample_rows(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    return frame.sample(n=max_rows, random_state=seed)


def _v_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col.startswith(V_COLUMN_PREFIX) and col[1:].isdigit()]


def _drop_high_missing(
    frame: pd.DataFrame, columns: list[str], max_missing_fraction: float
) -> list[str]:
    missing = frame[columns].isna().mean()
    return [col for col in columns if missing[col] <= max_missing_fraction]


def _label_association(corr: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    # |Spearman with the label| ranks a column's standalone monotonic fraud signal.
    label = corr[LABEL_COLUMN]
    return {col: 0.0 if pd.isna(label[col]) else abs(float(label[col])) for col in columns}


def _decorrelate(
    v_corr: pd.DataFrame, strength: dict[str, float], correlation_threshold: float
) -> list[str]:
    # Visit columns strongest-first; each becomes a leader that claims every still-free
    # column it correlates with above the threshold. Leaders are the representatives.
    ordered = sorted(v_corr.index, key=lambda col: (-strength[col], col))
    representatives: list[str] = []
    claimed: set[str] = set()
    for col in ordered:
        if col in claimed:
            continue
        representatives.append(col)
        claimed.update(v_corr.index[v_corr[col] >= correlation_threshold])
    return representatives


def _v_only_frame(path: Path) -> pd.DataFrame:
    # Read only the V columns plus the label; the full frame carries ~430 columns.
    names = pq.ParquetFile(path).schema.names
    v_columns = [col for col in names if col.startswith(V_COLUMN_PREFIX) and col[1:].isdigit()]
    return pd.read_parquet(path, columns=[*v_columns, LABEL_COLUMN])


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    params = load_params()
    train = _v_only_frame(PROCESSED_DIR / "train.parquet")
    selected = select_v_columns(train, params.select_v, params.seed)
    V_SELECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    V_SELECTED_PATH.write_text(json.dumps(selected, indent=2) + "\n")
    log.info("v_selected", candidates=len(_v_columns(train)), selected=len(selected))


if __name__ == "__main__":
    main()
