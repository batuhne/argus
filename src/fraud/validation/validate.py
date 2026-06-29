"""Validate the raw data against the pandera schemas and a fraud-rate gate; exit on failure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pandera.errors

from fraud.paths import RAW_DIR
from fraud.validation.schemas import IdentitySchema, TransactionSchema

REPORT_PATH = Path("reports") / "validation_report.json"
TRANSACTION_COLUMNS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card4",
    "card6",
    "C1",
    "C14",
    "D1",
    "D15",
    "dist1",
    "dist2",
    "addr1",
    "addr2",
]

# A target rate far outside this band means the wrong file or a corrupted label
# column, not real drift. Halt rather than feed it downstream.
MIN_FRAUD_RATE = 0.005
MAX_FRAUD_RATE = 0.15


def validate(raw_dir: Path = RAW_DIR, report_path: Path = REPORT_PATH) -> None:
    transactions = pd.read_csv(raw_dir / "train_transaction.csv", usecols=TRANSACTION_COLUMNS)
    identities = pd.read_csv(raw_dir / "train_identity.csv", usecols=["TransactionID"])

    try:
        TransactionSchema.validate(transactions, lazy=True)
        IdentitySchema.validate(identities, lazy=True)
    except pandera.errors.SchemaErrors as exc:
        sys.exit(f"validation failed:\n{exc}")

    fraud_rate = float(transactions["isFraud"].mean())
    if not MIN_FRAUD_RATE <= fraud_rate <= MAX_FRAUD_RATE:
        sys.exit(
            f"fraud rate {fraud_rate:.4f} is outside the expected "
            f"[{MIN_FRAUD_RATE}, {MAX_FRAUD_RATE}] range"
        )

    report = {
        "transaction_rows": len(transactions),
        "identity_rows": len(identities),
        "fraud_rate": round(fraud_rate, 6),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    validate()
