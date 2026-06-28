from pathlib import Path

import pytest
from pydantic import ValidationError

from fraud.params import CalibrationParams, Params, load_params


def test_load_params_exposes_training_and_evaluation_sections() -> None:
    params = load_params()
    assert params.training.candidate_alias == "candidate"
    assert params.training.optuna.n_trials == 50
    assert params.training.optuna.timeout_seconds == 3600
    assert params.training.encoder.smoothing == 20.0
    assert params.training.encoder.n_splits == 5
    assert params.training.shap.sample_size == 2000
    assert params.evaluation.champion_alias == "champion"
    assert params.evaluation.calibration.method == "isotonic"
    assert params.evaluation.threshold.recall_floor == 0.1
    assert params.evaluation.threshold.alert_volume_budget == 0.1
    assert params.evaluation.gate.auprc_tolerance == 0.0
    assert params.evaluation.gate.cost_tolerance == 0.0


def test_recall_at_k_levels_is_a_tuple() -> None:
    levels = load_params().evaluation.recall_at_k_levels
    assert isinstance(levels, tuple)
    assert levels == (0.005, 0.01, 0.05)


def test_unknown_top_level_key_is_rejected() -> None:
    raw = load_params().model_dump()
    raw["unexpected"] = 1
    with pytest.raises(ValidationError):
        Params.model_validate(raw)


def test_unknown_nested_key_is_rejected() -> None:
    raw = load_params().model_dump()
    raw["training"]["mystery"] = 1
    with pytest.raises(ValidationError):
        Params.model_validate(raw)


def test_unsupported_calibration_method_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CalibrationParams.model_validate({"method": "platt"})


def test_split_fractions_must_leave_training_rows() -> None:
    with pytest.raises(ValidationError):
        Params.model_validate(
            {
                "seed": 1,
                "data": {"sample_size": None},
                "split": {"val_fraction": 0.5, "test_fraction": 0.4, "holdout_fraction": 0.2},
            }
        )


def test_load_params_reads_file_values_not_defaults(tmp_path: Path) -> None:
    custom = tmp_path / "params.yaml"
    custom.write_text(
        "seed: 7\n"
        "data:\n  sample_size: 1234\n"
        "split:\n  val_fraction: 0.2\n  test_fraction: 0.1\n  holdout_fraction: 0.1\n"
        "training:\n  optuna:\n    n_trials: 11\n"
    )
    params = load_params(custom)
    assert params.seed == 7
    assert params.data.sample_size == 1234
    assert params.training.optuna.n_trials == 11


def test_load_params_missing_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"nope\.yaml"):
        load_params(tmp_path / "nope.yaml")


def test_load_params_empty_file_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "params.yaml"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty or not a mapping"):
        load_params(empty)
