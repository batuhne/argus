from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


def shap_summary_figure(model: Any, x_sample: pd.DataFrame, max_display: int = 20) -> Figure:
    """Beeswarm summary on a precomputed sample. Caller is responsible for sizing."""
    explainer = shap.TreeExplainer(model)
    explanation = explainer(x_sample)
    figure = plt.figure(figsize=(10, max(4, max_display * 0.35)))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    return figure


def feature_schema_payload(frame: pd.DataFrame) -> dict[str, dict[str, float | int | str | None]]:
    payload: dict[str, dict[str, float | int | str | None]] = {}
    for column, series in frame.items():
        payload[str(column)] = {
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "min": _scalar(series.min(skipna=True)),
            "max": _scalar(series.max(skipna=True)),
        }
    return payload


def _scalar(value: Any) -> float | int | str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, int | float):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)
