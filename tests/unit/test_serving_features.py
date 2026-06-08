import numpy as np
import pandas as pd
import pytest

from fraud.ingestion.stream import RawAttributes
from fraud.serving.features import assemble_features
from fraud.training.features import FEATURE_COLUMNS, LABEL_COLUMN
from fraud.transforms import feature_logic as fl
from fraud.transforms.encoders import CategoricalEncoder, fit_encoder


@pytest.fixture
def encoder() -> CategoricalEncoder:
    rng = np.random.default_rng(0)
    rows = 60
    frame = pd.DataFrame(
        {column: rng.choice(["a", "b", "c"], rows) for column in fl.CATEGORICAL_COLUMNS}
    )
    frame[LABEL_COLUMN] = rng.integers(0, 2, rows)
    return fit_encoder(frame, fl.CATEGORICAL_COLUMNS, LABEL_COLUMN)


def _online_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "card_id": "1_2_3_5",
                "card_txn_count_24h": 4.0,
                "card_amt_sum_24h": 400.0,
                "card_amt_mean_24h": 100.0,
                "card_amt_max_24h": 250.0,
                "seconds_since_prev_txn": 1800.0,
                "amt_to_card_mean_24h": 1.5,
                "amt_log": 5.0,
            }
        ]
    )


def test_assemble_orders_columns_to_model_contract(encoder: CategoricalEncoder) -> None:
    assembled = assemble_features(_online_row(), amount=150.0, raw=RawAttributes(), encoder=encoder)
    assert list(assembled.columns) == list(FEATURE_COLUMNS)


def test_assemble_injects_raw_transaction_amount(encoder: CategoricalEncoder) -> None:
    assembled = assemble_features(_online_row(), amount=150.0, raw=RawAttributes(), encoder=encoder)
    assert assembled["TransactionAmt"].iloc[0] == 150.0


def test_assemble_keeps_missing_raw_numeric_as_nan(encoder: CategoricalEncoder) -> None:
    assembled = assemble_features(
        _online_row(), amount=150.0, raw=RawAttributes(C1=7.0), encoder=encoder
    )
    assert assembled["C1"].iloc[0] == 7.0
    assert pd.isna(assembled["C13"].iloc[0])
    assert assembled["C1"].dtype == "float32"


def test_assemble_matches_offline_coercion_for_raw_numerics(encoder: CategoricalEncoder) -> None:
    raw = RawAttributes(C1=5.0, dist1=2.5, addr1=300.0)
    assembled = assemble_features(_online_row(), amount=150.0, raw=raw, encoder=encoder)

    offline = fl.coerce_numeric(pd.DataFrame([raw.model_dump()]), fl.RAW_NUMERIC_PASSTHROUGH)
    for column in fl.RAW_NUMERIC_PASSTHROUGH:
        np.testing.assert_array_equal(
            assembled[column].to_numpy(),
            offline[column].to_numpy(),
            err_msg=f"raw-numeric skew in {column}",
        )


def test_assemble_places_v_vector_and_fills_missing(encoder: CategoricalEncoder) -> None:
    selected = fl.V_SELECTED
    assert selected, "expected a frozen V set from select_v"

    assembled = assemble_features(
        _online_row(), amount=150.0, raw=RawAttributes(v={selected[0]: 0.75}), encoder=encoder
    )

    assert assembled[selected[0]].iloc[0] == pytest.approx(0.75)
    assert assembled[selected[0]].dtype == "float32"  # coerced like the offline path
    assert pd.isna(assembled[selected[-1]].iloc[0])  # a frozen V the request omitted


def test_assemble_encodes_a_present_categorical(encoder: CategoricalEncoder) -> None:
    cat = fl.CATEGORICAL_COLUMNS[0]
    assembled = assemble_features(
        _online_row(), amount=150.0, raw=RawAttributes(categorical={cat: "a"}), encoder=encoder
    )

    assert assembled[f"{cat}_freq"].dtype == "float32"
    assert assembled[f"{cat}_freq"].iloc[0] == pytest.approx(encoder.frequency_maps[cat]["a"])


def test_assemble_categoricals_match_offline_transform(encoder: CategoricalEncoder) -> None:
    cat_values: dict[str, str | None] = dict.fromkeys(fl.CATEGORICAL_COLUMNS, "a")
    offline = encoder.transform(pd.DataFrame([cat_values]))

    assembled = assemble_features(
        _online_row(), amount=150.0, raw=RawAttributes(categorical=cat_values), encoder=encoder
    )

    for column in offline.columns:
        assert assembled[column].iloc[0] == pytest.approx(offline[column].iloc[0]), column


def test_assemble_encodes_omitted_categorical_as_missing(encoder: CategoricalEncoder) -> None:
    none_values = dict.fromkeys(fl.CATEGORICAL_COLUMNS)
    offline_missing = encoder.transform(pd.DataFrame([none_values]))

    assembled = assemble_features(_online_row(), amount=150.0, raw=RawAttributes(), encoder=encoder)

    cat = fl.CATEGORICAL_COLUMNS[0]
    assert assembled[f"{cat}_target"].iloc[0] == pytest.approx(
        offline_missing[f"{cat}_target"].iloc[0]
    )


def test_assemble_does_not_mutate_input_frame(encoder: CategoricalEncoder) -> None:
    online = _online_row()
    assemble_features(online, amount=150.0, raw=RawAttributes(), encoder=encoder)
    assert "TransactionAmt" not in online.columns
    assert "C1" not in online.columns
