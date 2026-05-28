import pathlib
from collections.abc import Callable

import pandas as pd
import pytest

from fraud.training.train import TrainingConfig, train_with_splits

pytestmark = pytest.mark.integration

SyntheticSplit = Callable[[int, int], tuple[pd.DataFrame, pd.Series]]


def test_train_with_splits_registers_and_returns_metrics(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    make_synthetic_split: SyntheticSplit,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path / 'mlruns'}")
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", f"file://{tmp_path / 'mlruns'}")
    cfg = TrainingConfig(
        seed=42,
        tracking_uri=f"file://{tmp_path / 'mlruns'}",
        experiment_name="argus_test",
        model_name="argus_test_model",
        candidate_alias="candidate",
        optuna_n_trials=2,
        optuna_timeout=120,
        shap_sample_size=200,
        recall_at_k_levels=(0.05, 0.1),
        run_name="argus_test_run",
        repo_dir=tmp_path,
        processed_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
    )
    splits = {
        "train": make_synthetic_split(600, 1),
        "val": make_synthetic_split(200, 2),
        "test": make_synthetic_split(200, 3),
    }

    result = train_with_splits(cfg, splits)

    assert result.run_id
    assert result.model_version == 1
    assert result.primary.family in {"xgboost", "lightgbm"}
    assert result.primary.val_metrics["auprc"] > 0.7
    assert result.primary.test_metrics["auprc"] > 0.7
