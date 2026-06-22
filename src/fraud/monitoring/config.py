from __future__ import annotations

from dataclasses import dataclass

from fraud.config import get_settings
from fraud.evaluation.business import CostMatrix
from fraud.ingestion.stream import (
    DRIFT_ALERTS_TOPIC,
    LABELS_TOPIC,
    MONITOR_GROUP,
    SCORED_FEATURES_TOPIC,
)
from fraud.params import load_params


@dataclass(frozen=True, slots=True)
class MonitoringConfig:
    bootstrap_servers: str
    scored_features_topic: str
    labels_topic: str
    drift_alerts_topic: str
    consumer_group: str
    exporter_port: int
    tracking_uri: str
    model_name: str
    champion_alias: str
    cost_matrix: CostMatrix
    psi_threshold: float
    psi_top_n: int
    auprc_floor: float
    drift_debounce_cycles: int
    window_size: int
    join_retention: int
    recompute_interval_seconds: float
    min_matched_for_auprc: int
    min_current_for_drift: int

    @classmethod
    def from_settings(cls) -> MonitoringConfig:
        settings = get_settings()
        params = load_params()
        monitoring = params.monitoring
        cost = params.evaluation.cost_matrix
        return cls(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            scored_features_topic=SCORED_FEATURES_TOPIC,
            labels_topic=LABELS_TOPIC,
            drift_alerts_topic=DRIFT_ALERTS_TOPIC,
            consumer_group=MONITOR_GROUP,
            exporter_port=settings.monitoring_exporter_port,
            tracking_uri=settings.mlflow_tracking_uri,
            model_name=settings.argus_model_name,
            champion_alias="champion",
            cost_matrix=CostMatrix(fn_cost_usd=cost.fn_cost_usd, fp_cost_usd=cost.fp_cost_usd),
            psi_threshold=monitoring.psi_threshold,
            psi_top_n=monitoring.psi_top_n,
            auprc_floor=monitoring.auprc_floor,
            drift_debounce_cycles=monitoring.drift_debounce_cycles,
            window_size=monitoring.window_size,
            join_retention=monitoring.join_retention,
            recompute_interval_seconds=monitoring.recompute_interval_seconds,
            min_matched_for_auprc=monitoring.min_matched_for_auprc,
            min_current_for_drift=monitoring.min_current_for_drift,
        )
