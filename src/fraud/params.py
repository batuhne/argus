from pathlib import Path

import yaml
from pydantic import BaseModel, Field

PARAMS_FILE = Path("params.yaml")


class DataParams(BaseModel):
    sample_size: int | None = None


class SplitParams(BaseModel):
    val_fraction: float
    test_fraction: float


class CostMatrixParams(BaseModel):
    fn_cost_usd: float = 100.0
    fp_cost_usd: float = 5.0


class EvaluationParams(BaseModel):
    cost_matrix: CostMatrixParams = Field(default_factory=CostMatrixParams)


class MonitoringParams(BaseModel):
    psi_threshold: float = 0.2
    auprc_floor: float = 0.05
    drift_debounce_cycles: int = 3
    window_size: int = 5000
    join_retention: int = 20000
    recompute_interval_seconds: float = 15.0
    min_matched_for_auprc: int = 200
    min_current_for_drift: int = 500


class Params(BaseModel):
    seed: int
    data: DataParams
    split: SplitParams
    evaluation: EvaluationParams = Field(default_factory=EvaluationParams)
    monitoring: MonitoringParams = Field(default_factory=MonitoringParams)


def load_params(path: Path = PARAMS_FILE) -> Params:
    with path.open() as handle:
        raw = yaml.safe_load(handle)
    return Params.model_validate(raw)
