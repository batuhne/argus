"""Feature drift detection against a frozen training baseline via Evidently."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

# A PSI above this is the conventional threshold for a meaningful population shift.
DEFAULT_PSI_THRESHOLD = 0.2
PSI_METHOD = "psi"


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    psi: dict[str, float]
    psi_threshold: float

    @property
    def drifted_features(self) -> list[str]:
        return [name for name, value in self.psi.items() if value > self.psi_threshold]

    @property
    def max_psi(self) -> float:
        return max(self.psi.values(), default=0.0)

    @property
    def drift_share(self) -> float:
        return len(self.drifted_features) / len(self.psi) if self.psi else 0.0


def compute_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Sequence[str],
    *,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
) -> FeatureDrift:
    """Per-feature PSI of the current window against the frozen reference."""
    from evidently import Report
    from evidently.metrics import ValueDrift

    cols = list(columns)
    report = Report([ValueDrift(column=col, method=PSI_METHOD) for col in cols])
    snapshot = report.run(reference_data=reference[cols], current_data=current[cols])
    return FeatureDrift(psi=_extract_psi(snapshot.dict(), cols), psi_threshold=psi_threshold)


def build_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, columns: Sequence[str]
) -> Any:
    """Full Evidently drift snapshot (PSI, KS, Wasserstein) for the audit report."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    cols = list(columns)
    report = Report([DataDriftPreset()], include_tests=True)
    return report.run(reference_data=reference[cols], current_data=current[cols])


def _extract_psi(result: dict[str, Any], columns: Sequence[str]) -> dict[str, float]:
    wanted = set(columns)
    psi: dict[str, float] = {}
    for metric in result.get("metrics", []):
        config = metric.get("config", {})
        column = config.get("column")
        value = metric.get("value")
        if column in wanted and isinstance(value, int | float):
            psi[column] = float(value)
    return psi
