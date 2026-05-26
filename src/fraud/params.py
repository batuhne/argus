from pathlib import Path

import yaml
from pydantic import BaseModel

PARAMS_FILE = Path("params.yaml")


class DataParams(BaseModel):
    sample_size: int | None = None


class SplitParams(BaseModel):
    val_fraction: float
    test_fraction: float


class Params(BaseModel):
    seed: int
    data: DataParams
    split: SplitParams


def load_params(path: Path = PARAMS_FILE) -> Params:
    with path.open() as handle:
        raw = yaml.safe_load(handle)
    return Params.model_validate(raw)
