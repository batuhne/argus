from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


@dataclass(frozen=True, slots=True)
class BoostingHyperparams:
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    min_child_weight: float = 1.0
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50


def compute_scale_pos_weight(y: pd.Series) -> float:
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        warnings.warn(
            "scale_pos_weight undefined for single-class labels; falling back to 1.0",
            stacklevel=2,
        )
        return 1.0
    return float(negatives / positives)


def build_xgb(params: BoostingHyperparams, scale_pos_weight: float, seed: int) -> Any:
    return XGBClassifier(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        min_child_weight=params.min_child_weight,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        reg_alpha=params.reg_alpha,
        reg_lambda=params.reg_lambda,
        early_stopping_rounds=params.early_stopping_rounds,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
    )


def build_lgb(params: BoostingHyperparams, scale_pos_weight: float, seed: int) -> Any:
    return LGBMClassifier(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        min_child_weight=params.min_child_weight,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        reg_alpha=params.reg_alpha,
        reg_lambda=params.reg_lambda,
        objective="binary",
        metric="average_precision",
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
