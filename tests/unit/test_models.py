import pandas as pd
import pytest

from fraud.training.models import (
    BoostingHyperparams,
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
