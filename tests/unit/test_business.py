import math

import numpy as np
import pytest

from fraud.evaluation.business import (
    CostMatrix,
    confusion_at_threshold,
    expected_cost,
    expected_cost_per_transaction,
)


def test_cost_matrix_rejects_negative_fn_cost() -> None:
    with pytest.raises(ValueError, match="fn_cost_usd"):
        CostMatrix(fn_cost_usd=-1.0, fp_cost_usd=5.0)


def test_cost_matrix_rejects_negative_fp_cost() -> None:
    with pytest.raises(ValueError, match="fp_cost_usd"):
        CostMatrix(fn_cost_usd=100.0, fp_cost_usd=-0.01)


def test_confusion_at_threshold_partitions_correctly() -> None:
    y_true = np.array([1, 0, 1, 0, 1])
    y_score = np.array([0.9, 0.1, 0.4, 0.8, 0.6])

    tp, fp, tn, fn = confusion_at_threshold(y_true, y_score, threshold=0.5)

    assert (tp, fp, tn, fn) == (2, 1, 1, 1)


def test_expected_cost_weights_components_by_matrix() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.2, 0.8, 0.1])
    matrix = CostMatrix(fn_cost_usd=100.0, fp_cost_usd=5.0)

    cost = expected_cost(y_true, y_score, threshold=0.5, matrix=matrix)

    assert cost == pytest.approx(1 * 100.0 + 1 * 5.0)


def test_expected_cost_per_transaction_returns_nan_for_empty() -> None:
    matrix = CostMatrix(fn_cost_usd=100.0, fp_cost_usd=5.0)
    result = expected_cost_per_transaction(np.array([]), np.array([]), 0.5, matrix)
    assert math.isnan(result)


def test_expected_cost_raises_on_shape_mismatch() -> None:
    matrix = CostMatrix(fn_cost_usd=100.0, fp_cost_usd=5.0)
    with pytest.raises(ValueError, match="align"):
        expected_cost(np.array([1, 0, 1]), np.array([0.5, 0.5]), 0.5, matrix)
