import re
from pathlib import Path
from typing import Any

import yaml

from fraud.params import load_params

ALERTS_FILE = Path(__file__).resolve().parents[2] / "infra" / "prometheus" / "alerts.yml"


def _rules() -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(ALERTS_FILE.read_text())
    return {rule["alert"]: rule for group in data["groups"] for rule in group["rules"]}


def _number(pattern: str, expr: str) -> float:
    match = re.search(pattern, expr)
    assert match is not None, f"pattern {pattern!r} not found in {expr!r}"
    return float(match.group(1))


def test_feature_drift_alert_threshold_matches_psi_threshold() -> None:
    expr = _rules()["FeatureDriftDetected"]["expr"]
    threshold = _number(r"argus_feature_drift_psi_max\s*>\s*([0-9.]+)", expr)
    assert threshold == load_params().monitoring.psi_threshold


def test_performance_decay_alert_thresholds_match_params() -> None:
    expr = _rules()["ModelPerformanceDecay"]["expr"]
    monitoring = load_params().monitoring
    assert _number(r"argus_rolling_auprc\s*<\s*([0-9.]+)", expr) == monitoring.auprc_floor
    assert _number(r"argus_matched_join\s*>\s*([0-9]+)", expr) == monitoring.min_matched_for_auprc
