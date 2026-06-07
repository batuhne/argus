import pandas as pd
import pandera.errors
import pytest

from fraud.validation.schemas import IdentitySchema, TransactionSchema


def _valid_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "isFraud": [0, 1, 0],
            "TransactionDT": [86400, 86500, 90000],
            "TransactionAmt": [50.0, 125.5, 12.0],
            "ProductCD": ["W", "C", "H"],
            "C1": [1.0, 2.0, 3.0],
            "C14": [0.0, 1.0, 2.0],
            "D1": [10.0, None, 5.0],
            "D15": [0.0, 1.0, None],
            "dist1": [1.0, None, 2.0],
            "dist2": [None, 1.0, 2.0],
            "addr1": [100.0, 200.0, 300.0],
            "addr2": [10.0, 20.0, 30.0],
            "V1": [1.0, None, 3.0],
        }
    )


def test_schema_accepts_valid_transactions() -> None:
    TransactionSchema.validate(_valid_transactions())


def test_schema_allows_unconstrained_columns() -> None:
    df = _valid_transactions()
    assert "V1" in df.columns
    TransactionSchema.validate(df)


def test_schema_rejects_target_outside_binary() -> None:
    df = _valid_transactions()
    df.loc[0, "isFraud"] = 2
    with pytest.raises(pandera.errors.SchemaError):
        TransactionSchema.validate(df)


def test_schema_rejects_unknown_product_code() -> None:
    df = _valid_transactions()
    df.loc[0, "ProductCD"] = "Z"
    with pytest.raises(pandera.errors.SchemaError):
        TransactionSchema.validate(df)


def test_identity_schema_requires_transaction_id() -> None:
    df = pd.DataFrame({"id_01": [1.0, 2.0]})
    with pytest.raises(pandera.errors.SchemaError):
        IdentitySchema.validate(df)
