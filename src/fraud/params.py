from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PARAMS_FILE = Path("params.yaml")


class _StrictModel(BaseModel):
    # An unknown key in params.yaml is config drift; fail loudly instead of dropping it.
    model_config = ConfigDict(extra="forbid")


class DataParams(_StrictModel):
    sample_size: int | None = Field(default=None, gt=0)


class SplitParams(_StrictModel):
    val_fraction: float = Field(gt=0.0, lt=1.0)
    test_fraction: float = Field(gt=0.0, lt=1.0)
    holdout_fraction: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _leaves_training_rows(self) -> "SplitParams":
        total = self.val_fraction + self.test_fraction + self.holdout_fraction
        if total >= 1.0:
            raise ValueError(f"val + test + holdout = {total} leaves no training rows")
        return self


class SelectVParams(_StrictModel):
    max_missing_fraction: float = Field(default=0.9, ge=0.0, lt=1.0)
    correlation_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_features: int = Field(default=30, ge=1)
    # Spearman is computed on a seeded row sample; correlation structure stabilizes
    # long before the full train split, and this keeps the offline stage minutes-fast.
    corr_sample_rows: int = Field(default=100_000, ge=1)


class OptunaParams(_StrictModel):
    n_trials: int = Field(default=50, ge=1)
    timeout_seconds: int = Field(default=3600, gt=0)


class EncoderParams(_StrictModel):
    smoothing: float = Field(default=20.0, gt=0.0)
    # StratifiedKFold needs at least two folds for out-of-fold target encoding.
    n_splits: int = Field(default=5, ge=2)


class ShapParams(_StrictModel):
    sample_size: int = Field(default=2000, ge=1)


class TrainingParams(_StrictModel):
    candidate_alias: str = "candidate"
    optuna: OptunaParams = Field(default_factory=OptunaParams)
    encoder: EncoderParams = Field(default_factory=EncoderParams)
    shap: ShapParams = Field(default_factory=ShapParams)


class CostMatrixParams(_StrictModel):
    fn_cost_usd: float = Field(default=100.0, ge=0.0)
    fp_cost_usd: float = Field(default=5.0, ge=0.0)


class CalibrationParams(_StrictModel):
    # Only isotonic is implemented today.
    method: Literal["isotonic"] = "isotonic"


class ThresholdParams(_StrictModel):
    recall_floor: float = Field(default=0.1, ge=0.0, le=1.0)
    alert_volume_budget: float = Field(default=0.1, ge=0.0, le=1.0)


class GateParams(_StrictModel):
    auprc_tolerance: float = Field(default=0.0, ge=0.0)
    cost_tolerance: float = Field(default=0.0, ge=0.0)


class EvaluationParams(_StrictModel):
    champion_alias: str = "champion"
    cost_matrix: CostMatrixParams = Field(default_factory=CostMatrixParams)
    recall_at_k_levels: tuple[float, ...] = (0.005, 0.01, 0.05)
    calibration: CalibrationParams = Field(default_factory=CalibrationParams)
    threshold: ThresholdParams = Field(default_factory=ThresholdParams)
    gate: GateParams = Field(default_factory=GateParams)


class MonitoringParams(_StrictModel):
    psi_threshold: float = Field(default=0.2, gt=0.0)
    psi_top_n: int = Field(default=15, ge=1)
    auprc_floor: float = Field(default=0.30, ge=0.0, le=1.0)
    drift_debounce_cycles: int = Field(default=3, ge=1)
    window_size: int = Field(default=5000, ge=1)
    retention_seconds: float = Field(default=1800.0, gt=0.0)
    recompute_interval_seconds: float = Field(default=15.0, gt=0.0)
    min_matched_for_auprc: int = Field(default=200, ge=1)
    min_current_for_drift: int = Field(default=500, ge=1)


class RetrainingParams(_StrictModel):
    deployment_name: str = "argus-retraining/argus-retraining"
    cron: str = "0 3 * * 1"
    # A retrain is expensive, so back-to-back drift alerts inside this window are
    # collapsed into one run to avoid a retraining storm.
    cooldown_seconds: float = Field(default=3600.0, ge=0.0)
    n_trials: int = Field(default=30, ge=1)
    timeout_seconds: int = Field(default=1800, gt=0)


class CanaryParams(_StrictModel):
    traffic_steps: tuple[float, ...] = (0.05, 0.25, 1.0)
    max_p99_latency_ms: float = Field(default=50.0, gt=0.0)
    max_error_rate: float = Field(default=0.001, ge=0.0, le=1.0)
    # The canary's AUPRC (once delayed labels arrive) must stay within this
    # fraction of the live champion before it earns more traffic.
    min_auprc_ratio: float = Field(default=0.98, gt=0.0, le=1.0)
    # Online guard available before labels: canary and champion must agree on
    # this fraction of decisions over the same mirrored traffic.
    min_agreement: float = Field(default=0.95, ge=0.0, le=1.0)
    min_samples: int = Field(default=500, ge=1)
    max_holds_per_step: int = Field(default=3, ge=0)


class StreamParams(_StrictModel):
    # Dataset-seconds replayed per real second; compresses the multi-week holdout into a
    # watchable run and divides the chargeback lag by the same factor.
    time_warp_factor: float = Field(default=2000.0, gt=0.0)
    # Real-world chargeback lag before a label lands, warped down by time_warp_factor.
    base_chargeback_lag_days: float = Field(default=7.0, ge=0.0)
    # Fractional spread on each transaction's lag; bounded below 1 to keep the lag positive.
    label_lag_jitter: float = Field(default=0.2, ge=0.0, lt=1.0)
    # Cap on any single inter-message wait so a long real-world gap cannot stall the replay.
    max_message_delay_seconds: float = Field(default=2.0, ge=0.0)


class Params(_StrictModel):
    seed: int
    data: DataParams
    split: SplitParams
    select_v: SelectVParams = Field(default_factory=SelectVParams)
    training: TrainingParams = Field(default_factory=TrainingParams)
    evaluation: EvaluationParams = Field(default_factory=EvaluationParams)
    monitoring: MonitoringParams = Field(default_factory=MonitoringParams)
    retraining: RetrainingParams = Field(default_factory=RetrainingParams)
    canary: CanaryParams = Field(default_factory=CanaryParams)
    stream: StreamParams = Field(default_factory=StreamParams)


def load_params(path: Path = PARAMS_FILE) -> Params:
    try:
        text = path.read_text()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"params file not found at {path.resolve()}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"params file at {path.resolve()} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"params file at {path.resolve()} is empty or not a mapping")
    return Params.model_validate(raw)
