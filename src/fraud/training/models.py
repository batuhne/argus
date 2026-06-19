from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier
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
    # Minimum loss reduction to split. XGBoost calls it gamma; LightGBM, min_split_gain.
    gamma: float = 0.0
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
        gamma=params.gamma,
        early_stopping_rounds=params.early_stopping_rounds,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        max_bin=64,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=2,
    )


def build_lgb(params: BoostingHyperparams, scale_pos_weight: float, seed: int) -> Any:
    # LightGBM grows leaf-wise; cap num_leaves to a full binary tree at max_depth
    # so its per-tree capacity matches XGBoost's level-wise growth at the same depth.
    num_leaves = max(2, 2**params.max_depth - 1)
    return LGBMClassifier(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        num_leaves=num_leaves,
        learning_rate=params.learning_rate,
        min_child_weight=params.min_child_weight,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        reg_alpha=params.reg_alpha,
        reg_lambda=params.reg_lambda,
        min_split_gain=params.gamma,
        objective="binary",
        metric="average_precision",
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=2,
        verbose=-1,
    )


def build_cat(params: BoostingHyperparams, scale_pos_weight: float, seed: int) -> Any:
    # CatBoost trains on the same encoded numeric matrix as the others, so its native
    # categorical handling is unused and the serving contract stays single. reg_alpha,
    # gamma, and min_child_weight have no CatBoost analogue and are dropped.
    # allow_writing_files keeps it from littering a catboost_info dir each run.
    return CatBoostClassifier(
        iterations=params.n_estimators,
        depth=params.max_depth,
        learning_rate=params.learning_rate,
        l2_leaf_reg=params.reg_lambda,
        subsample=params.subsample,
        rsm=params.colsample_bytree,
        early_stopping_rounds=params.early_stopping_rounds,
        eval_metric="PRAUC",
        scale_pos_weight=scale_pos_weight,
        random_seed=seed,
        thread_count=2,
        allow_writing_files=False,
        verbose=False,
    )
