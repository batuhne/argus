import pytest
from pipelines.flows.monitoring_pipeline import validated_split

from fraud.training.dataset import SPLITS


@pytest.mark.parametrize("split", SPLITS)
def test_validated_split_accepts_known_splits(split: str) -> None:
    assert validated_split(split) == split


@pytest.mark.parametrize("bad", ["../secret", "prod", "train.parquet", ""])
def test_validated_split_rejects_unknown(bad: str) -> None:
    with pytest.raises(ValueError, match="current_split must be one of"):
        validated_split(bad)
