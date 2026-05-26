import pandas as pd
import pytest

from fraud.data.split import time_split


def test_time_split_is_chronological_and_complete() -> None:
    df = pd.DataFrame(
        {
            "TransactionDT": [5, 1, 4, 2, 3, 6, 8, 7, 9, 10],
            "isFraud": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        }
    )

    train, val, test = time_split(df, val_fraction=0.2, test_fraction=0.2)

    assert (len(train), len(val), len(test)) == (6, 2, 2)
    assert len(train) + len(val) + len(test) == len(df)
    assert train["TransactionDT"].max() < val["TransactionDT"].min()
    assert val["TransactionDT"].max() < test["TransactionDT"].min()


def test_time_split_rejects_boundary_time_overlap() -> None:
    # The same TransactionDT lands on both sides of the train/val boundary.
    df = pd.DataFrame({"TransactionDT": [1, 2, 3, 4, 5, 6, 6, 7, 8, 9]})
    with pytest.raises(ValueError, match="overlap"):
        time_split(df, val_fraction=0.2, test_fraction=0.2)
