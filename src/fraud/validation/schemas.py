import pandera.pandas as pa
from pandera.typing import Series

# IEEE-CIS transactions carry hundreds of anonymized V, C, D, and M columns. The
# schema pins the load-bearing fields (target, time, amount, product) as a data
# contract and leaves the rest unconstrained, so a column rename or a corrupted
# target fails the pipeline early.

PRODUCT_CODES = ["W", "H", "C", "S", "R"]


class TransactionSchema(pa.DataFrameModel):
    transaction_id: Series[int] = pa.Field(alias="TransactionID", unique=True, ge=0)
    is_fraud: Series[int] = pa.Field(alias="isFraud", isin=[0, 1])
    transaction_dt: Series[int] = pa.Field(alias="TransactionDT", gt=0)
    transaction_amt: Series[float] = pa.Field(alias="TransactionAmt", gt=0)
    product_cd: Series[str] = pa.Field(alias="ProductCD", isin=PRODUCT_CODES)

    class Config:
        strict = False
        coerce = True


class IdentitySchema(pa.DataFrameModel):
    transaction_id: Series[int] = pa.Field(alias="TransactionID", unique=True, ge=0)

    class Config:
        strict = False
        coerce = True
