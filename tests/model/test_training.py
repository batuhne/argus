import pathlib
from collections.abc import Callable

import pandas as pd
import pytest

from fraud.evaluation.backtest import _verify_recorded_champion
from fraud.evaluation.business import CostMatrix
from fraud.evaluation.gate import (
    GATE_BOOTSTRAP,
    GATE_COST_REGRESSION,
    GateMetrics,
    GateTolerances,
)
from fraud.evaluation.threshold import ThresholdConstraints
from fraud.registry import (
    CHAMPION_TAG_AUPRC,
    CHAMPION_TAG_COST_PER_TX,
    get_alias_version,
    get_version_tags,
)
from fraud.training.train import TrainingConfig, _write_run_marker, train_with_splits
from fraud.transforms.encoders import CategoricalEncoder

pytestmark = pytest.mark.integration

SyntheticSplit = Callable[[int, int], tuple[pd.DataFrame, pd.Series]]


def _stub_encoder() -> CategoricalEncoder:
    # The synthetic splits already carry the encoded columns as numeric features, so the
    # gate-logic tests only need an encoder object to log, not a fitted one.
    return CategoricalEncoder(columns=(), frequency_maps={}, target_maps={}, global_prior=0.0)


def _config(tmp_path: pathlib.Path) -> TrainingConfig:
    return TrainingConfig(
        seed=42,
        tracking_uri=f"file://{tmp_path / 'mlruns'}",
        experiment_name="argus_test",
        model_name="argus_test_model",
        candidate_alias="candidate",
        champion_alias="champion",
        optuna_n_trials=2,
        optuna_timeout=120,
        encoder_smoothing=20.0,
        encoder_n_splits=5,
        shap_sample_size=200,
        recall_at_k_levels=(0.05, 0.1),
        calibration_method="isotonic",
        cost_matrix=CostMatrix(fn_cost_usd=100.0, fp_cost_usd=5.0),
        threshold_constraints=ThresholdConstraints(recall_floor=0.5, alert_volume_budget=0.6),
        gate_tolerances=GateTolerances(),
        run_name="argus_test_run",
        repo_dir=tmp_path,
        processed_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
    )


def _make_splits(
    factory: SyntheticSplit,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    return {
        "train": factory(600, 1),
        "val": factory(200, 2),
        "test": factory(200, 3),
    }


def test_train_with_splits_bootstraps_champion_on_first_run(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    make_synthetic_split: SyntheticSplit,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path / 'mlruns'}")
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", f"file://{tmp_path / 'mlruns'}")
    cfg = _config(tmp_path)
    splits = _make_splits(make_synthetic_split)

    result = train_with_splits(cfg, splits, _stub_encoder())

    assert result.model_version == 1
    assert result.champion_version == 1
    assert result.primary.val_metrics["auprc"] > 0.7
    assert result.gate.decision.promote
    assert result.gate.decision.reason == GATE_BOOTSTRAP
    assert get_alias_version(cfg.model_name, cfg.champion_alias) == 1
    tags = get_version_tags(cfg.model_name, 1)
    assert CHAMPION_TAG_AUPRC in tags
    assert CHAMPION_TAG_COST_PER_TX in tags

    _write_run_marker(cfg.artifacts_dir, result)
    marker = cfg.artifacts_dir / "last_run.json"
    _verify_recorded_champion(result.champion_version, marker)
    with pytest.raises(RuntimeError, match="alias moved"):
        _verify_recorded_champion(result.champion_version + 1, marker)


def test_gate_blocks_promotion_when_challenger_cost_regresses(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    make_synthetic_split: SyntheticSplit,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path / 'mlruns'}")
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", f"file://{tmp_path / 'mlruns'}")
    cfg = _config(tmp_path)
    splits = _make_splits(make_synthetic_split)

    first_run = train_with_splits(cfg, splits, _stub_encoder())

    fake_champion = GateMetrics(
        auprc=first_run.primary.test_metrics["auprc"] + 0.05,
        expected_cost_per_tx=max(first_run.gate.test_expected_cost_per_tx - 1.0, 0.0),
    )
    from fraud import registry

    registry.write_version_tags(
        cfg.model_name,
        1,
        {
            CHAMPION_TAG_AUPRC: f"{fake_champion.auprc:.6f}",
            CHAMPION_TAG_COST_PER_TX: f"{fake_champion.expected_cost_per_tx:.6f}",
        },
    )

    second_run = train_with_splits(cfg, splits, _stub_encoder())

    assert second_run.model_version == 2
    assert second_run.champion_version == 1
    assert second_run.gate.decision.promote is False
    assert second_run.gate.decision.reason in {GATE_COST_REGRESSION, "both_regression"}
    assert get_alias_version(cfg.model_name, cfg.champion_alias) == 1
