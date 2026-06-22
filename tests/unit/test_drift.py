import numpy as np
import pandas as pd

from fraud.monitoring.drift import _extract_psi, compute_feature_drift

COLUMNS = ["stable", "shifted"]


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    reference = pd.DataFrame(
        {"stable": rng.normal(0.0, 1.0, 3000), "shifted": rng.normal(5.0, 2.0, 3000)}
    )
    current = pd.DataFrame(
        {"stable": rng.normal(0.0, 1.0, 3000), "shifted": rng.normal(9.0, 2.0, 3000)}
    )
    return reference, current


def test_compute_feature_drift_flags_only_the_shifted_feature() -> None:
    reference, current = _frames()
    drift = compute_feature_drift(reference, current, COLUMNS)

    assert set(drift.psi) == set(COLUMNS)
    assert drift.psi["shifted"] > drift.psi_threshold
    assert drift.psi["stable"] < drift.psi_threshold
    assert drift.drifted_features == ["shifted"]
    assert drift.drift_share == 0.5
    assert drift.max_psi == drift.psi["shifted"]


def test_compute_feature_drift_clean_when_distributions_match() -> None:
    reference, _ = _frames()
    drift = compute_feature_drift(reference, reference.copy(), COLUMNS)

    assert drift.drifted_features == []
    assert drift.max_psi < drift.psi_threshold


def test_extract_psi_drops_non_finite_values() -> None:
    snapshot = {
        "metrics": [
            {"config": {"column": "stable"}, "value": 0.3},
            {"config": {"column": "broken"}, "value": float("nan")},
            {"config": {"column": "huge"}, "value": float("inf")},
        ]
    }

    psi = _extract_psi(snapshot, ["stable", "broken", "huge"])

    assert psi == {"stable": 0.3}
