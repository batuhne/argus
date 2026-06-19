import pandas as pd
import pytest

from fraud.training.models import (
    BoostingHyperparams,
    build_cat,
    build_lgb,
    build_xgb,
    compute_scale_pos_weight,
)


def test_scale_pos_weight_matches_neg_over_pos() -> None:
    y = pd.Series([0] * 99 + [1])
    assert compute_scale_pos_weight(y) == pytest.approx(99.0)


def test_scale_pos_weight_falls_back_when_no_positives() -> None:
    with pytest.warns(UserWarning, match="single-class"):
        assert compute_scale_pos_weight(pd.Series([0, 0, 0])) == 1.0


def test_scale_pos_weight_falls_back_when_no_negatives() -> None:
    with pytest.warns(UserWarning, match="single-class"):
        assert compute_scale_pos_weight(pd.Series([1, 1, 1])) == 1.0


def test_build_xgb_propagates_seed_and_imbalance() -> None:
    model = build_xgb(BoostingHyperparams(n_estimators=10), scale_pos_weight=12.5, seed=7)

    assert model.random_state == 7
    assert model.scale_pos_weight == 12.5
    assert model.eval_metric == "aucpr"
    assert model.tree_method == "hist"
    assert model.early_stopping_rounds == 50


def test_build_lgb_propagates_seed_and_imbalance() -> None:
    model = build_lgb(BoostingHyperparams(n_estimators=10), scale_pos_weight=12.5, seed=7)

    assert model.random_state == 7
    assert model.scale_pos_weight == 12.5
    assert model.metric == "average_precision"
    assert model.objective == "binary"


def test_build_lgb_num_leaves_matches_depth_capacity() -> None:
    shallow = build_lgb(BoostingHyperparams(max_depth=3), scale_pos_weight=1.0, seed=0)
    deep = build_lgb(BoostingHyperparams(max_depth=10), scale_pos_weight=1.0, seed=0)

    assert shallow.num_leaves == 7
    assert deep.num_leaves == 1023


def test_gamma_maps_to_each_family_split_penalty() -> None:
    xgb = build_xgb(BoostingHyperparams(gamma=2.5), scale_pos_weight=1.0, seed=0)
    lgb = build_lgb(BoostingHyperparams(gamma=2.5), scale_pos_weight=1.0, seed=0)

    assert xgb.gamma == 2.5
    assert lgb.min_split_gain == 2.5


def test_build_cat_propagates_seed_and_imbalance() -> None:
    model = build_cat(BoostingHyperparams(n_estimators=10), scale_pos_weight=12.5, seed=7)
    params = model.get_params()

    assert params["random_seed"] == 7
    assert params["scale_pos_weight"] == 12.5
    assert params["eval_metric"] == "PRAUC"
    # No catboost_info dir written during training/test runs.
    assert params["allow_writing_files"] is False
