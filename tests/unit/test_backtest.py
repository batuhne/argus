import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud.evaluation.backtest import _verify_recorded_champion, evaluate_holdout
from fraud.evaluation.business import CostMatrix


def _matrix() -> CostMatrix:
    return CostMatrix(fn_cost_usd=100.0, fp_cost_usd=5.0)


def test_evaluate_holdout_assembles_metrics_at_threshold() -> None:
    # Perfectly separable: the two positives score highest, threshold flags exactly them.
    y = pd.Series([0, 0, 0, 1, 1], dtype="int8")
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])

    report = evaluate_holdout(y, scores, threshold=0.5, cost_matrix=_matrix(), recall_levels=(0.4,))

    assert report.rows == 5
    assert report.positives == 2
    assert report.auprc == pytest.approx(1.0)
    assert report.precision == pytest.approx(1.0)
    assert report.recall == pytest.approx(1.0)
    assert report.flagged_rate == pytest.approx(0.4)
    assert report.expected_cost_total_usd == pytest.approx(0.0)
    assert report.expected_cost_per_tx_usd == pytest.approx(0.0)
    # Top 40% = top 2 rows = both positives.
    assert report.recall_at_k[0.4] == pytest.approx(1.0)
    # brier = mean squared residual = (0.01 + 0.04 + 0.09 + 0.04 + 0.01) / 5
    assert report.brier == pytest.approx(0.038)


def test_evaluate_holdout_costs_missed_fraud_above_every_score() -> None:
    y = pd.Series([0, 0, 1, 1], dtype="int8")
    scores = np.array([0.1, 0.2, 0.3, 0.4])

    # Threshold above every score: nothing flagged, both frauds missed.
    report = evaluate_holdout(y, scores, threshold=0.9, cost_matrix=_matrix(), recall_levels=(0.5,))

    assert report.flagged_rate == pytest.approx(0.0)
    assert report.recall == pytest.approx(0.0)
    # Two false negatives at $100 each, spread over four transactions.
    assert report.expected_cost_total_usd == pytest.approx(200.0)
    assert report.expected_cost_per_tx_usd == pytest.approx(50.0)


def _write_marker(path: Path, champion_version: object) -> None:
    path.write_text(json.dumps({"champion_version": champion_version}))


def test_verify_recorded_champion_passes_when_versions_match(tmp_path: Path) -> None:
    marker = tmp_path / "last_run.json"
    _write_marker(marker, 7)

    _verify_recorded_champion(7, marker)


def test_verify_recorded_champion_raises_when_alias_drifted(tmp_path: Path) -> None:
    marker = tmp_path / "last_run.json"
    _write_marker(marker, 7)

    with pytest.raises(RuntimeError, match="alias moved"):
        _verify_recorded_champion(8, marker)


def test_verify_recorded_champion_skips_when_marker_absent(tmp_path: Path) -> None:
    _verify_recorded_champion(8, tmp_path / "missing.json")


def test_verify_recorded_champion_skips_when_version_unrecorded(tmp_path: Path) -> None:
    marker = tmp_path / "last_run.json"
    _write_marker(marker, None)

    _verify_recorded_champion(8, marker)
