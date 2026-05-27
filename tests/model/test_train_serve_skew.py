"""Training-serving skew guard.

Asserts the offline (point-in-time joined) features match the online (Redis)
features for the same card, to floating-point tolerance.
"""

from __future__ import annotations

import socket
import uuid
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from feast import FeatureStore

from fraud.features.registry import build_objects
from fraud.transforms import feature_logic as fl

pytestmark = pytest.mark.integration

REDIS_HOST = "localhost"
REDIS_PORT = 6379
UTC_BASE = pd.Timestamp("2017-12-01", tz="UTC")

# Per card: the last entry is the one the online store ends up holding.
TRANSACTIONS = pd.DataFrame(
    {
        "card_id": ["A", "A", "A", "B", "B"],
        "event_timestamp": UTC_BASE + pd.to_timedelta([0, 3600, 7200, 0, 1800], unit="s"),
        "TransactionAmt": [10.0, 20.0, 30.0, 7.0, 9.0],
    }
)
LATEST = pd.DataFrame(
    {
        "card_id": ["A", "B"],
        "event_timestamp": UTC_BASE + pd.to_timedelta([7200, 1800], unit="s"),
        "TransactionAmt": [30.0, 9.0],
    }
)
FEATURE_COLUMNS = [*fl.VELOCITY_COLUMNS, "amt_to_card_mean_24h", "amt_log"]


def _redis_reachable() -> bool:
    try:
        with socket.create_connection((REDIS_HOST, REDIS_PORT), timeout=1):
            return True
    except OSError:
        return False


def _write_repo(tmp_path: Path, features_path: Path) -> Path:
    project = f"skew_{uuid.uuid4().hex[:8]}"
    (tmp_path / "feature_store.yaml").write_text(
        f"project: {project}\n"
        "provider: local\n"
        "registry: registry.db\n"
        "offline_store:\n"
        "  type: file\n"
        "online_store:\n"
        "  type: redis\n"
        f"  connection_string: {REDIS_HOST}:{REDIS_PORT}\n"
        "entity_key_serialization_version: 3\n"
    )
    return features_path


@pytest.fixture
def store(tmp_path: Path) -> Iterator[FeatureStore]:
    if not _redis_reachable():
        pytest.skip("redis is not reachable on localhost:6379")

    features_path = tmp_path / "card_features.parquet"
    fl.compute_card_velocity(TRANSACTIONS).to_parquet(features_path, index=False)
    _write_repo(tmp_path, features_path)

    feature_store = FeatureStore(repo_path=str(tmp_path))
    feature_store.apply(build_objects(features_path).to_list())
    feature_store.materialize(
        start_date=(UTC_BASE - timedelta(days=1)).to_pydatetime(),
        end_date=(UTC_BASE + timedelta(days=1)).to_pydatetime(),
    )
    try:
        yield feature_store
    finally:
        feature_store.teardown()  # type: ignore[no-untyped-call]


def test_online_features_match_offline(store: FeatureStore) -> None:
    service = store.get_feature_service("card_activity")

    offline = (
        store.get_historical_features(entity_df=LATEST, features=service)
        .to_df()
        .set_index("card_id")
        .loc[LATEST["card_id"]]
    )
    entity_rows = [
        {"card_id": card_id, "TransactionAmt": amount}
        for card_id, amount in zip(LATEST["card_id"], LATEST["TransactionAmt"], strict=True)
    ]
    online = (
        store.get_online_features(features=service, entity_rows=entity_rows)
        .to_df()
        .set_index("card_id")
        .loc[LATEST["card_id"]]
    )

    for column in FEATURE_COLUMNS:
        np.testing.assert_allclose(
            online[column].to_numpy(dtype="float64"),
            offline[column].to_numpy(dtype="float64"),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"training-serving skew in {column}",
        )
