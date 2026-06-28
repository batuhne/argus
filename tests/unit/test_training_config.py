import pytest

from fraud.params import load_params
from fraud.training.train import TrainingConfig


def test_from_settings_reflects_params(monkeypatch: pytest.MonkeyPatch) -> None:
    base = load_params()
    tuned = base.model_copy(
        update={
            "training": base.training.model_copy(
                update={
                    "candidate_alias": "candidate_marker",
                    "optuna": base.training.optuna.model_copy(update={"n_trials": 123}),
                }
            ),
            "evaluation": base.evaluation.model_copy(
                update={
                    "champion_alias": "champion_marker",
                    "threshold": base.evaluation.threshold.model_copy(
                        update={"recall_floor": 0.37}
                    ),
                }
            ),
        }
    )
    monkeypatch.setattr("fraud.training.train.load_params", lambda: tuned)

    cfg = TrainingConfig.from_settings()

    assert cfg.candidate_alias == "candidate_marker"
    assert cfg.champion_alias == "champion_marker"
    assert cfg.optuna_n_trials == 123
    assert cfg.threshold_constraints.recall_floor == 0.37


def test_from_settings_honors_overrides() -> None:
    cfg = TrainingConfig.from_settings(n_trials=7, timeout=9)
    assert cfg.optuna_n_trials == 7
    assert cfg.optuna_timeout == 9
