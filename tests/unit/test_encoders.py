from pathlib import Path

import pandas as pd
import pytest

from fraud.transforms.encoders import (
    encoded_feature_names,
    fit_encoder,
    fit_transform_oof,
    load_encoder,
    save_encoder,
)
from fraud.transforms.features import LABEL_COLUMN


def test_encoded_feature_names_match_transform_output_order() -> None:
    frame = pd.DataFrame({"x": ["a", "b"], "y": ["c", "d"], LABEL_COLUMN: [0, 1]})

    out = fit_encoder(frame, ["x", "y"], LABEL_COLUMN).transform(frame)

    assert encoded_feature_names(["x", "y"]) == tuple(out.columns)


def test_transform_outputs_named_float32_columns() -> None:
    frame = pd.DataFrame({"cat": ["a", "b"], LABEL_COLUMN: [0, 1]})

    out = fit_encoder(frame, ["cat"], LABEL_COLUMN).transform(frame)

    assert list(out.columns) == ["cat_freq", "cat_target"]
    assert out["cat_freq"].dtype == "float32"
    assert out["cat_target"].dtype == "float32"


def test_frequency_encoding_is_category_proportion() -> None:
    frame = pd.DataFrame({"cat": ["a", "a", "b", "c"], LABEL_COLUMN: [0, 1, 0, 1]})

    out = fit_encoder(frame, ["cat"], LABEL_COLUMN).transform(frame)

    assert out["cat_freq"].iloc[0] == pytest.approx(0.5)  # 'a' is 2 of 4
    assert out["cat_freq"].iloc[2] == pytest.approx(0.25)  # 'b' is 1 of 4


def test_target_encoding_applies_smoothing() -> None:
    # prior = 3/4; 'a' has count 2, mean 0.5; smoothed = (2*0.5 + 2*0.75) / (2+2) = 0.625
    frame = pd.DataFrame({"cat": ["a", "a", "b", "b"], LABEL_COLUMN: [0, 1, 1, 1]})

    out = fit_encoder(frame, ["cat"], LABEL_COLUMN, smoothing=2.0).transform(frame)

    assert out["cat_target"].iloc[0] == pytest.approx(0.625, abs=1e-6)


def test_unseen_category_falls_back_to_prior() -> None:
    frame = pd.DataFrame({"cat": ["a", "a", "b", "c"], LABEL_COLUMN: [0, 1, 0, 1]})
    encoder = fit_encoder(frame, ["cat"], LABEL_COLUMN, smoothing=0.0)

    out = encoder.transform(pd.DataFrame({"cat": ["unseen"]}))

    assert out["cat_freq"].iloc[0] == 0.0
    assert out["cat_target"].iloc[0] == pytest.approx(encoder.global_prior)


def test_missing_is_its_own_learned_category() -> None:
    frame = pd.DataFrame({"cat": ["a", "a", "b", None], LABEL_COLUMN: [0, 1, 0, 1]})
    encoder = fit_encoder(frame, ["cat"], LABEL_COLUMN, smoothing=0.0)

    out = encoder.transform(pd.DataFrame({"cat": [None]}))

    # The single missing row was seen in fit, so NaN maps to its learned frequency, not 0.
    assert out["cat_freq"].iloc[0] == pytest.approx(0.25)


def _singleton_frame() -> pd.DataFrame:
    # A balanced frame where one fraud row carries a category seen nowhere else.
    cats = ["common"] * 40
    cats[20] = "rare"
    return pd.DataFrame({"cat": cats, LABEL_COLUMN: [0] * 20 + [1] * 20})


def test_out_of_fold_target_excludes_the_rows_own_label() -> None:
    frame = _singleton_frame()

    full = fit_encoder(frame, ["cat"], LABEL_COLUMN, smoothing=0.0).transform(frame)
    _, oof = fit_transform_oof(frame, ["cat"], LABEL_COLUMN, seed=0, smoothing=0.0, n_splits=5)

    # Full-train encoding sees the singleton's own label (1.0); out-of-fold must not.
    assert full["cat_target"].iloc[20] == pytest.approx(1.0)
    assert oof["cat_target"].iloc[20] < 1.0
    assert 0.0 <= oof["cat_target"].iloc[20] <= 1.0


def test_keys_are_stable_across_int_and_float_spelling() -> None:
    # Fit sees float64 (a NaN forced the upcast); serve sees int64. The same code must
    # map to the same key, or every value would silently fall back to the prior.
    fit_frame = pd.DataFrame({"code": [1.0, 2.0, None, 1.0], LABEL_COLUMN: [0, 1, 0, 1]})
    encoder = fit_encoder(fit_frame, ["code"], LABEL_COLUMN, smoothing=0.0)

    out = encoder.transform(pd.DataFrame({"code": pd.Series([1, 2], dtype="int64")}))

    assert out["code_freq"].iloc[0] == pytest.approx(0.5)  # code 1 seen 2 of 4
    assert out["code_freq"].iloc[1] == pytest.approx(0.25)  # code 2 seen 1 of 4


def test_fractional_float_categories_stay_distinct_without_crashing() -> None:
    # A categorical that arrives as a fractional float must not crash the int cast, and 13.5
    # must stay a different category from 13.0 (which collapses to the "13" spelling).
    frame = pd.DataFrame({"code": [13.5, 13.0, 13.0, 13.0], LABEL_COLUMN: [0, 1, 0, 1]})

    encoder = fit_encoder(frame, ["code"], LABEL_COLUMN, smoothing=0.0)
    out = encoder.transform(frame)

    assert set(encoder.frequency_maps["code"]) == {"13", "13.5"}
    assert out["code_freq"].iloc[0] == pytest.approx(0.25)  # 13.5 seen 1 of 4
    assert out["code_freq"].iloc[1] == pytest.approx(0.75)  # 13.0 -> "13" seen 3 of 4


def test_non_finite_and_out_of_range_floats_do_not_crash_the_int_cast() -> None:
    # inf and floats past the int64 range have no integer spelling; they become their own
    # string category rather than raising in the cast.
    frame = pd.DataFrame(
        {"code": [float("inf"), float("-inf"), 1e19, 7.0], LABEL_COLUMN: [0, 1, 0, 1]}
    )

    encoder = fit_encoder(frame, ["code"], LABEL_COLUMN, smoothing=0.0)
    out = encoder.transform(frame)

    assert set(encoder.frequency_maps["code"]) == {"inf", "-inf", "1e+19", "7"}
    assert out["code_freq"].notna().all()


def test_out_of_fold_preserves_a_non_default_index() -> None:
    frame = _singleton_frame()
    frame.index = pd.RangeIndex(100, 140)

    _, oof = fit_transform_oof(frame, ["cat"], LABEL_COLUMN, seed=0, smoothing=0.0, n_splits=5)

    assert list(oof.index) == list(range(100, 140))
    assert oof["cat_target"].notna().all()
    assert oof["cat_target"].loc[120] < 1.0  # the singleton row, own label excluded


def test_out_of_fold_rejects_n_splits_above_minority_count() -> None:
    frame = pd.DataFrame({"cat": ["a"] * 6, LABEL_COLUMN: [0, 0, 0, 0, 0, 1]})

    with pytest.raises(ValueError, match="n_splits"):
        fit_transform_oof(frame, ["cat"], LABEL_COLUMN, seed=0, n_splits=5)


def test_out_of_fold_is_deterministic() -> None:
    frame = _singleton_frame()

    _, first = fit_transform_oof(frame, ["cat"], LABEL_COLUMN, seed=7)
    _, second = fit_transform_oof(frame, ["cat"], LABEL_COLUMN, seed=7)

    pd.testing.assert_frame_equal(first, second)


def test_encoder_survives_joblib_round_trip(tmp_path: Path) -> None:
    frame = pd.DataFrame({"cat": ["a", "a", "b", "c"], LABEL_COLUMN: [0, 1, 0, 1]})
    encoder = fit_encoder(frame, ["cat"], LABEL_COLUMN)
    path = tmp_path / "encoder.joblib"

    save_encoder(encoder, path)
    loaded = load_encoder(path)

    pd.testing.assert_frame_equal(encoder.transform(frame), loaded.transform(frame))
