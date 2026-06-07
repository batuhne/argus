import numpy as np
import pandas as pd

from fraud.ingestion.stream import RawAttributes
from fraud.serving.features import assemble_features
from fraud.training.features import FEATURE_COLUMNS
from fraud.transforms import feature_logic as fl


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


def test_assemble_orders_columns_to_model_contract() -> None:
    assembled = assemble_features(_online_row(), amount=150.0, raw=RawAttributes())
    assert list(assembled.columns) == list(FEATURE_COLUMNS)


def test_assemble_injects_raw_transaction_amount() -> None:
    assembled = assemble_features(_online_row(), amount=150.0, raw=RawAttributes())
    assert assembled["TransactionAmt"].iloc[0] == 150.0


def test_assemble_keeps_missing_raw_numeric_as_nan() -> None:
    assembled = assemble_features(_online_row(), amount=150.0, raw=RawAttributes(C1=7.0))
    assert assembled["C1"].iloc[0] == 7.0
    assert pd.isna(assembled["C13"].iloc[0])
    assert assembled["C1"].dtype == "float32"


def test_assemble_matches_offline_coercion_for_raw_numerics() -> None:
    raw = RawAttributes(C1=5.0, dist1=2.5, addr1=300.0)
    assembled = assemble_features(_online_row(), amount=150.0, raw=raw)

    offline = fl.coerce_numeric(pd.DataFrame([raw.model_dump()]), fl.RAW_NUMERIC_PASSTHROUGH)
    for column in fl.RAW_NUMERIC_PASSTHROUGH:
        np.testing.assert_array_equal(
            assembled[column].to_numpy(),
            offline[column].to_numpy(),
            err_msg=f"raw-numeric skew in {column}",
        )


def test_assemble_does_not_mutate_input_frame() -> None:
    online = _online_row()
    assemble_features(online, amount=150.0, raw=RawAttributes())
    assert "TransactionAmt" not in online.columns
    assert "C1" not in online.columns
