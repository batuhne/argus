import pathlib

import numpy as np
import pandas as pd
import pytest

from fraud.training.features import FEATURE_COLUMNS, LABEL_COLUMN
from fraud.training.train import TrainingConfig, train_with_splits

pytestmark = pytest.mark.integration


def _synthetic_split(rows: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, size=rows)
    base = rng.normal(size=(rows, len(FEATURE_COLUMNS)))
    signal = label[:, None] * 3.0
    x = pd.DataFrame(base + signal, columns=list(FEATURE_COLUMNS))
    y = pd.Series(label, name=LABEL_COLUMN, dtype="int8")
    return x, y


def test_train_with_splits_registers_and_returns_metrics(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
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
        "train": _synthetic_split(rows=600, seed=1),
        "val": _synthetic_split(rows=200, seed=2),
        "test": _synthetic_split(rows=200, seed=3),
    }

    result = train_with_splits(cfg, splits)

    assert result.run_id
    assert result.model_version == 1
    assert result.primary.family in {"xgboost", "lightgbm"}
    assert result.primary.val_metrics["auprc"] > 0.7
    assert result.primary.test_metrics["auprc"] > 0.7
