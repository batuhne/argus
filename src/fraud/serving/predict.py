"""Score one assembled feature row with the champion bundle into a calibrated decision."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fraud.model_loader import ModelBundle


@dataclass(frozen=True, slots=True)
class ScoredTransaction:
    fraud_score: float
    decision: bool
    threshold: float


def score_transaction(bundle: ModelBundle, features: pd.DataFrame) -> ScoredTransaction:
    """Calibrated fraud probability and flag decision for one assembled feature row."""
    raw = float(bundle.model.predict_proba(features)[:, 1][0])
    calibrated = float(bundle.calibrator.predict([raw])[0])
    return ScoredTransaction(
        fraud_score=calibrated,
        decision=calibrated >= bundle.threshold,
        threshold=bundle.threshold,
    )
