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


class RetrainingParams(BaseModel):
    deployment_name: str = "argus-retraining/argus-retraining"
    cron: str = "0 3 * * 1"
    # A retrain is expensive, so back-to-back drift alerts inside this window are
    # collapsed into one run to avoid a retraining storm.
    cooldown_seconds: float = 3600.0
    n_trials: int = 30
    timeout_seconds: int = 1800


class CanaryParams(BaseModel):
    traffic_steps: tuple[float, ...] = (0.05, 0.25, 1.0)
    max_p99_latency_ms: float = 50.0
    max_error_rate: float = 0.001
    # The canary's AUPRC (once delayed labels arrive) must stay within this
    # fraction of the live champion before it earns more traffic.
    min_auprc_ratio: float = 0.98
    # Online guard available before labels: canary and champion must agree on
    # this fraction of decisions over the same mirrored traffic.
    min_agreement: float = 0.95
    min_samples: int = 500
    max_holds_per_step: int = 3


class Params(BaseModel):
    seed: int
    data: DataParams
    split: SplitParams
    evaluation: EvaluationParams = Field(default_factory=EvaluationParams)
    monitoring: MonitoringParams = Field(default_factory=MonitoringParams)
    retraining: RetrainingParams = Field(default_factory=RetrainingParams)
    canary: CanaryParams = Field(default_factory=CanaryParams)


def load_params(path: Path = PARAMS_FILE) -> Params:
    with path.open() as handle:
        raw = yaml.safe_load(handle)
    return Params.model_validate(raw)
