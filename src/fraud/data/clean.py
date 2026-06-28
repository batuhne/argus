from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud.params import Params, load_params
from fraud.paths import INTERIM_DIR, RAW_DIR

OUTPUT_PATH = INTERIM_DIR / "clean.parquet"


def merge_transactions(transactions: pd.DataFrame, identities: pd.DataFrame) -> pd.DataFrame:
    # validate="1:1" asserts TransactionID is unique on both sides, so a
    # duplicated or malformed key fails here instead of silently fanning out rows.
    return transactions.merge(identities, on="TransactionID", how="left", validate="1:1")


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink the wide float and int columns in place so the frame fits in memory."""
    float_cols = list(df.select_dtypes("float64").columns)
    if float_cols:
        df[float_cols] = df[float_cols].astype("float32")
    for col in df.select_dtypes("int64").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def stratified_sample(df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Sample down for a faster dev loop while keeping the fraud rate intact."""
    if size >= len(df):
        return df
    fraction = size / len(df)
    return df.groupby("isFraud", group_keys=False).sample(frac=fraction, random_state=seed)


def clean(
    raw_dir: Path = RAW_DIR,
    output_path: Path = OUTPUT_PATH,
    params: Params | None = None,
) -> None:
    resolved = params or load_params()
    transactions = downcast(pd.read_csv(raw_dir / "train_transaction.csv"))
    identities = downcast(pd.read_csv(raw_dir / "train_identity.csv"))

    merged = merge_transactions(transactions, identities)
    if resolved.data.sample_size is not None:
        merged = stratified_sample(merged, resolved.data.sample_size, resolved.seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)


if __name__ == "__main__":
    clean()
