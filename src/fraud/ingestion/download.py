"""Fetch the IEEE-CIS fraud detection data from Kaggle.

This is a setup step run by hand or as the first pipeline stage. It needs
KAGGLE_API_TOKEN in the environment (see .env.example) and the competition rules
accepted at https://www.kaggle.com/competitions/ieee-fraud-detection/rules.

Only the labeled training files are fetched. The competition test set has no
public labels, so validation and test splits are built from the training data.
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

from fraud.paths import RAW_DIR

COMPETITION = "ieee-fraud-detection"
FILES = ("train_transaction.csv", "train_identity.csv")


def download(raw_dir: Path = RAW_DIR) -> None:
    load_dotenv()
    if not os.environ.get("KAGGLE_API_TOKEN"):
        sys.exit("KAGGLE_API_TOKEN is not set. Copy .env.example to .env and add your token.")

    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if (raw_dir / name).exists():
            continue
        _fetch(name, raw_dir)
        _unzip(name, raw_dir)


def _fetch(name: str, raw_dir: Path) -> None:
    cmd = ["kaggle", "competitions", "download", "-c", COMPETITION, "-f", name, "-p", str(raw_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = result.stderr or result.stdout
        if "403" in message or "Forbidden" in message:
            sys.exit(
                "Kaggle returned 403 for the competition data. Accept the rules at "
                "https://www.kaggle.com/competitions/ieee-fraud-detection/rules and retry."
            )
        sys.exit(f"kaggle download failed for {name}:\n{message}")


def _unzip(name: str, raw_dir: Path) -> None:
    archive = raw_dir / f"{name}.zip"
    if not archive.exists():
        return
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)
    archive.unlink()


if __name__ == "__main__":
    download()
