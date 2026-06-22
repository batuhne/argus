import pytest
from pydantic import ValidationError

from fraud.monitoring.config import MonitoringConfig
from fraud.params import MonitoringParams


def test_monitoring_params_reject_non_positive_psi_top_n() -> None:
    with pytest.raises(ValidationError):
        MonitoringParams(psi_top_n=0)


def test_monitoring_config_loads_topics_and_thresholds() -> None:
    cfg = MonitoringConfig.from_settings()

    assert cfg.scored_features_topic == "scored-features"
    assert cfg.labels_topic == "labels"
    assert cfg.drift_alerts_topic == "drift-alerts"
    assert cfg.consumer_group == "argus-fraud-monitor"
    assert cfg.psi_threshold == 0.2
    assert cfg.psi_top_n == 15
    assert cfg.auprc_floor == 0.30
    assert cfg.cost_matrix.fn_cost_usd == 100.0
    assert cfg.cost_matrix.fp_cost_usd == 5.0
    assert cfg.drift_debounce_cycles >= 1
    assert cfg.window_size > 0
