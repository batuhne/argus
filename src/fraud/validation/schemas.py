import pandera.pandas as pa
from pandera.typing import Series

# IEEE-CIS transactions carry hundreds of anonymized V, C, D, M columns. The schema
# pins the load-bearing fields plus one boundary anchor per model-input family (C, D,
# dist, addr); dataset.py presence-checks the rest by name from the split parquet.

PRODUCT_CODES = ["W", "H", "C", "S", "R"]


class TransactionSchema(pa.DataFrameModel):
    transaction_id: Series[int] = pa.Field(alias="TransactionID", unique=True, ge=0)
    is_fraud: Series[int] = pa.Field(alias="isFraud", isin=[0, 1])
    transaction_dt: Series[int] = pa.Field(alias="TransactionDT", gt=0)
    transaction_amt: Series[float] = pa.Field(alias="TransactionAmt", gt=0)
    product_cd: Series[str] = pa.Field(alias="ProductCD", isin=PRODUCT_CODES)
    c1: Series[float] = pa.Field(alias="C1", nullable=True)
    c14: Series[float] = pa.Field(alias="C14", nullable=True)
    d1: Series[float] = pa.Field(alias="D1", nullable=True)
    d15: Series[float] = pa.Field(alias="D15", nullable=True)
    dist1: Series[float] = pa.Field(alias="dist1", nullable=True)
    dist2: Series[float] = pa.Field(alias="dist2", nullable=True)
    addr1: Series[float] = pa.Field(alias="addr1", nullable=True)
    addr2: Series[float] = pa.Field(alias="addr2", nullable=True)

    class Config:
        strict = False
        coerce = True


class IdentitySchema(pa.DataFrameModel):
    transaction_id: Series[int] = pa.Field(alias="TransactionID", unique=True, ge=0)

    class Config:
        strict = False
        coerce = True
