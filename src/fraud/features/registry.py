"""Feast registry objects.

Built by a factory so the production repo and the skew test register identical
views over different Parquet files. The on-demand view reuses feature_logic, the
same module the offline builder uses.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    RequestSource,
    ValueType,
)
from feast.feast_object import FeastObject
from feast.on_demand_feature_view import OnDemandFeatureView, on_demand_feature_view
from feast.types import Float32, Float64

from fraud.paths import CARD_FEATURES_PATH
from fraud.transforms import feature_logic as fl


class RepoObjects(NamedTuple):
    card: Entity
    card_velocity: FeatureView
    transaction_dynamics: OnDemandFeatureView
    card_activity: FeatureService

    def to_list(self) -> list[FeastObject]:
        objects: list[FeastObject] = [
            self.card,
            self.card_velocity,
            self.transaction_dynamics,
            self.card_activity,
        ]
        return objects


def build_objects(source_path: Path) -> RepoObjects:
    card = Entity(name="card", join_keys=["card_id"], value_type=ValueType.STRING)

    source = FileSource(
        name="card_features_source",
        path=str(source_path.resolve()),
        timestamp_field="event_timestamp",
    )

    card_velocity = FeatureView(
        name="card_velocity",
        entities=[card],
        ttl=timedelta(days=365),
        schema=[Field(name=name, dtype=Float32) for name in fl.VELOCITY_COLUMNS],
        source=source,
        online=True,
    )

    txn_request = RequestSource(
        name="txn_request",
        schema=[Field(name="TransactionAmt", dtype=Float64)],
    )

    @on_demand_feature_view(  # type: ignore[untyped-decorator]
        sources=[card_velocity, txn_request],
        schema=[
            Field(name="amt_to_card_mean_24h", dtype=Float64),
            Field(name="amt_log", dtype=Float64),
        ],
        mode="pandas",
    )
    def transaction_dynamics(inputs: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=inputs.index)
        out["amt_to_card_mean_24h"] = fl.amount_to_mean_ratio(
            inputs["TransactionAmt"], inputs["card_amt_mean_24h"]
        )
        out["amt_log"] = fl.amount_log(inputs["TransactionAmt"])
        return out

    card_activity = FeatureService(
        name="card_activity",
        features=[card_velocity, transaction_dynamics],
    )

    return RepoObjects(card, card_velocity, transaction_dynamics, card_activity)


def default_objects() -> RepoObjects:
    return build_objects(CARD_FEATURES_PATH)
