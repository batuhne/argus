import pandas as pd

from fraud.data.clean import downcast, merge_transactions, stratified_sample


def test_merge_keeps_every_transaction() -> None:
    transactions = pd.DataFrame({"TransactionID": [1, 2, 3], "isFraud": [0, 1, 0]})
    identities = pd.DataFrame({"TransactionID": [2], "id_01": [9.0]})

    merged = merge_transactions(transactions, identities)

    assert len(merged) == 3
    assert "id_01" in merged.columns
    assert pd.isna(merged.loc[merged["TransactionID"] == 1, "id_01"]).all()


def test_downcast_shrinks_float_columns() -> None:
    df = pd.DataFrame({"v": pd.Series([1.0, 2.0, 3.0], dtype="float64")})
    out = downcast(df)
    assert str(out["v"].dtype) == "float32"


def test_stratified_sample_preserves_fraud_rate() -> None:
    df = pd.DataFrame({"isFraud": [0] * 950 + [1] * 50, "x": range(1000)})

    sampled = stratified_sample(df, size=200, seed=42)

    assert len(sampled) == 200
    assert abs(float(sampled["isFraud"].mean()) - 0.05) < 0.005
