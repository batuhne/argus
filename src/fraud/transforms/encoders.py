"""Categorical encoders: frequency plus smoothed, out-of-fold target encoding fit on train only."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# A missing categorical is encoded as its own category: in IEEE-CIS the absence is itself signal.
MISSING = "__missing__"
FREQ_SUFFIX = "_freq"
TARGET_SUFFIX = "_target"
DEFAULT_SMOOTHING = 20.0
DEFAULT_N_SPLITS = 5


@dataclass(frozen=True, slots=True)
class CategoricalEncoder:
    """Frozen frequency and smoothed-target maps, fit on train, applied identically everywhere."""

    columns: tuple[str, ...]
    frequency_maps: dict[str, dict[str, float]]
    target_maps: dict[str, dict[str, float]]
    global_prior: float

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Encode each categorical column into a frequency and a target column."""
        encoded: dict[str, pd.Series] = {}
        for column in self.columns:
            keys = _as_keys(frame[column])
            encoded[f"{column}{FREQ_SUFFIX}"] = keys.map(self.frequency_maps[column]).fillna(0.0)
            encoded[f"{column}{TARGET_SUFFIX}"] = keys.map(self.target_maps[column]).fillna(
                self.global_prior
            )
        return pd.DataFrame(encoded, index=frame.index).astype("float32")


def fit_encoder(
    frame: pd.DataFrame,
    columns: Sequence[str],
    label: str,
    smoothing: float = DEFAULT_SMOOTHING,
) -> CategoricalEncoder:
    """Fit the full-train maps used for val, test, and serving."""
    prior = float(frame[label].mean())
    frequency = {col: _frequency_map(_as_keys(frame[col])) for col in columns}
    target = {
        col: _target_map(_as_keys(frame[col]), frame[label], prior, smoothing) for col in columns
    }
    return CategoricalEncoder(tuple(columns), frequency, target, prior)


def fit_transform_oof(
    frame: pd.DataFrame,
    columns: Sequence[str],
    label: str,
    *,
    seed: int,
    smoothing: float = DEFAULT_SMOOTHING,
    n_splits: int = DEFAULT_N_SPLITS,
) -> tuple[CategoricalEncoder, pd.DataFrame]:
    """Return the encoder to persist for serving, plus the out-of-fold matrix to train on.

    Out-of-fold is what stops target encoding from leaking: a row never sees its own label.
    """
    _require_enough_per_class(frame[label], n_splits)
    encoder = fit_encoder(frame, columns, label, smoothing)
    encoded: dict[str, pd.Series] = {
        f"{col}{FREQ_SUFFIX}": _as_keys(frame[col]).map(encoder.frequency_maps[col]).fillna(0.0)
        for col in columns
    }
    encoded.update(_oof_target_columns(frame, columns, label, smoothing, n_splits, seed))
    return encoder, pd.DataFrame(encoded, index=frame.index).astype("float32")


def save_encoder(encoder: CategoricalEncoder, path: Path) -> None:
    joblib.dump(encoder, path)


def load_encoder(path: Path) -> CategoricalEncoder:
    # Loaded only from our own MLflow run artifact (trusted provenance), never from user input;
    # the encoder holds plain dicts and floats, so the pickle surface carries no callables.
    encoder: CategoricalEncoder = joblib.load(path)
    return encoder


def _oof_target_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    label: str,
    smoothing: float,
    n_splits: int,
    seed: int,
) -> dict[str, pd.Series]:
    y = frame[label]
    # An integer seed (not a RandomState instance) keeps the folds identical across runs.
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    columns_out: dict[str, pd.Series] = {
        f"{col}{TARGET_SUFFIX}": pd.Series(np.nan, index=frame.index, dtype="float64")
        for col in columns
    }
    for fit_idx, hold_idx in folds.split(frame, y):
        fold_prior = float(y.iloc[fit_idx].mean())
        for col in columns:
            fold_map = _target_map(
                _as_keys(frame[col].iloc[fit_idx]), y.iloc[fit_idx], fold_prior, smoothing
            )
            hold_values = _as_keys(frame[col].iloc[hold_idx]).map(fold_map).fillna(fold_prior)
            columns_out[f"{col}{TARGET_SUFFIX}"].iloc[hold_idx] = hold_values.to_numpy()
    return columns_out


def _require_enough_per_class(y: pd.Series, n_splits: int) -> None:
    # Each fold must contain both classes, or the out-of-fold target collapses to a
    # degenerate prior; fail loudly instead of silently degrading the encoding.
    smallest_class = int(y.value_counts().min())
    if smallest_class < n_splits:
        raise ValueError(
            f"n_splits={n_splits} exceeds the smallest class count {smallest_class}; "
            "out-of-fold target encoding needs each class present in every fold"
        )


def _as_keys(values: pd.Series) -> pd.Series:
    # Route numerics through Int64 first so "13" and "13.0" hash to the same category. One
    # stray NaN at fit time upcasts the column to float, and then train and serve disagree.
    if pd.api.types.is_integer_dtype(values) or pd.api.types.is_float_dtype(values):
        keys = values.astype("Int64").astype("string")
    else:
        keys = values.astype("string")
    return keys.where(values.notna(), MISSING).astype(str)


def _frequency_map(keys: pd.Series) -> dict[str, float]:
    total = float(len(keys))
    return {str(key): count / total for key, count in keys.value_counts().items()}


def _target_map(keys: pd.Series, y: pd.Series, prior: float, smoothing: float) -> dict[str, float]:
    stats = y.groupby(keys).agg(["mean", "count"])
    smoothed = (stats["count"] * stats["mean"] + smoothing * prior) / (stats["count"] + smoothing)
    return {str(key): float(value) for key, value in smoothed.items()}
