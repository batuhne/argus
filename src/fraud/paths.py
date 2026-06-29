"""Repository path constants for data, the feature repo, and the feature service name."""

from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

FEATURE_REPO_DIR = Path("feature_repo")
FEATURE_DATA_DIR = FEATURE_REPO_DIR / "data"
CARD_FEATURES_PATH = FEATURE_DATA_DIR / "card_features.parquet"
FEATURE_SERVICE = "card_activity"
