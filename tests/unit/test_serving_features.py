import pandas as pd

from fraud.serving.features import assemble_features
from fraud.training.features import FEATURE_COLUMNS


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
    assembled = assemble_features(_online_row(), amount=150.0)
    assert list(assembled.columns) == list(FEATURE_COLUMNS)


def test_assemble_injects_raw_transaction_amount() -> None:
    assembled = assemble_features(_online_row(), amount=150.0)
    assert assembled["TransactionAmt"].iloc[0] == 150.0


def test_assemble_does_not_mutate_input_frame() -> None:
    online = _online_row()
    assemble_features(online, amount=150.0)
    assert "TransactionAmt" not in online.columns
