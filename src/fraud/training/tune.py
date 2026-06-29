"""Tune XGBoost with an Optuna sweep on validation AUPRC."""

from __future__ import annotations

import gc
import logging

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial, Trial, TrialState

from fraud.common.logging import get_logger
from fraud.evaluation.metrics import auprc
from fraud.training.models import (
    BoostingHyperparams,
    build_xgb,
    compute_scale_pos_weight,
)

log = get_logger(__name__)


def tune_xgb(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    n_trials: int,
    timeout: int | None,
    seed: int,
) -> tuple[BoostingHyperparams, float]:
    """Maximize val AUPRC over the XGB search space; raises if no trial completes."""
    optuna.logging.get_logger("optuna").setLevel(logging.WARNING)
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    scale_pos_weight = compute_scale_pos_weight(y_train)

    def objective(trial: Trial) -> float:
        params = _sample_params(trial)
        model = build_xgb(params, scale_pos_weight=scale_pos_weight, seed=seed)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        scores = model.predict_proba(x_val)[:, 1]
        value = auprc(y_val, scores)
        del model, scores
        gc.collect()
        return value

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        catch=(ValueError, MemoryError),
        callbacks=[_log_trial],
    )

    completed = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("no successful Optuna trials; cannot select best params")

    return _params_from_trial(study.best_trial), float(study.best_value)


def _log_trial(study: Study, trial: FrozenTrial) -> None:
    best = study.best_value if study.best_trial is not None else float("nan")
    log.info(
        "optuna_trial",
        number=trial.number,
        state=trial.state.name,
        value=trial.value,
        best_so_far=best,
    )


def _sample_params(trial: Trial) -> BoostingHyperparams:
    # n_estimators reaches high; early stopping on the val set trims each fit to its
    # useful length, so the sampler spends its budget on shape, not tree count.
    return BoostingHyperparams(
        n_estimators=trial.suggest_int("n_estimators", 400, 2000),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        min_child_weight=trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
    )


def _params_from_trial(trial: FrozenTrial) -> BoostingHyperparams:
    params = trial.params
    return BoostingHyperparams(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        min_child_weight=float(params["min_child_weight"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        gamma=float(params["gamma"]),
    )
