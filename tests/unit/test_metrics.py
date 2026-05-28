import math

import numpy as np
import pandas as pd
import pytest

from fraud.evaluation.metrics import (
    auprc,
    classification_at_threshold,
    pr_curve_figure,
    recall_at_k,
)


def test_auprc_perfect_ranking_is_one() -> None:
    y_true = [0, 0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.8, 0.9]
    assert auprc(y_true, y_score) == pytest.approx(1.0)


def test_auprc_constant_score_equals_prevalence() -> None:
    y_true = np.array([1] * 20 + [0] * 80)
    y_score = np.full(100, 0.5)
    assert auprc(y_true, y_score) == pytest.approx(0.2)


def test_auprc_undefined_when_single_class() -> None:
    assert math.isnan(auprc([0, 0, 0], [0.1, 0.2, 0.3]))
    assert math.isnan(auprc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_recall_at_k_returns_one_when_k_covers_all_positives() -> None:
    y_true = [0] * 90 + [1] * 10
    y_score = list(np.linspace(0.0, 1.0, 100))
    assert recall_at_k(y_true, y_score, 0.1) == pytest.approx(1.0)


def test_recall_at_k_is_monotonic_in_k() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_score = rng.random(200)
    recalls = [recall_at_k(y_true, y_score, k) for k in (0.05, 0.1, 0.25, 0.5, 1.0)]
    assert all(recalls[i] <= recalls[i + 1] for i in range(len(recalls) - 1))
    assert recalls[-1] == pytest.approx(1.0)


def test_recall_at_k_handles_invalid_inputs() -> None:
    assert math.isnan(recall_at_k([0, 0, 1], [0.1, 0.2, 0.3], 0.0))
    assert math.isnan(recall_at_k([0, 0, 1], [0.1, 0.2, 0.3], 1.5))
    assert math.isnan(recall_at_k([0, 0, 0], [0.1, 0.2, 0.3], 0.5))


def test_classification_at_threshold_known_outcome() -> None:
    y_true = [1, 1, 0, 0, 1, 0]
    y_score = [0.9, 0.8, 0.7, 0.4, 0.3, 0.2]
    out = classification_at_threshold(y_true, y_score, threshold=0.75)
    assert out["precision"] == pytest.approx(1.0)
    assert out["recall"] == pytest.approx(2 / 3)
    assert out["false_positive_rate"] == pytest.approx(0.0)
    assert out["flagged_rate"] == pytest.approx(2 / 6)


def test_pr_curve_figure_dimensions_match_request() -> None:
    figure = pr_curve_figure([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert figure.get_size_inches().tolist() == [6.0, 5.0]
    assert len(figure.axes) == 1


def test_pr_curve_figure_handles_single_class_input() -> None:
    figure = pr_curve_figure([0, 0, 0], [0.1, 0.2, 0.3])
    assert figure.axes[0].title.get_text() == "PR curve"


def test_coerce_pair_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        auprc([0, 1], [0.1, 0.2, 0.3])


def test_metrics_accept_pandas_series() -> None:
    y_true = pd.Series([0, 1, 0, 1])
    y_score = pd.Series([0.2, 0.9, 0.3, 0.8])
    assert auprc(y_true, y_score) == pytest.approx(1.0)
