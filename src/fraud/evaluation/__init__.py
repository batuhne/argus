# reports.py is intentionally not re-exported here: it pulls shap/matplotlib,
# which the serving and monitoring runtimes must not require. Import it directly
# from fraud.evaluation.reports in the training code that needs it.
from fraud.evaluation.metrics import (
    auprc,
    classification_at_threshold,
    pr_curve_figure,
    recall_at_k,
)

__all__ = [
    "auprc",
    "classification_at_threshold",
    "pr_curve_figure",
    "recall_at_k",
]
