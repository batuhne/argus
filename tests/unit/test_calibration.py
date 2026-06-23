import numpy as np
import pytest

from fraud.evaluation.calibration import fit_isotonic, reliability_curve_figure


def test_fit_isotonic_lowers_brier_on_fit_data() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=2000)
    overconfident = np.clip(y * 0.95 + rng.normal(scale=0.1, size=2000), 0.0, 1.0)

    result = fit_isotonic(y, overconfident)

    assert result.brier_after <= result.brier_before


def test_isotonic_predict_clamps_out_of_range_input() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    in_range = rng.uniform(size=200)

    result = fit_isotonic(y, in_range)
    extreme = np.array([-0.5, 0.0, 0.5, 1.0, 1.8])
    calibrated = result.calibrator.predict(extreme)

    assert calibrated.min() >= 0.0
    assert calibrated.max() <= 1.0


def test_fit_isotonic_raises_on_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="align"):
        fit_isotonic(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))


@pytest.mark.parametrize("label", [0, 1])
def test_fit_isotonic_raises_on_single_class(label: int) -> None:
    y = np.full(20, label, dtype=int)
    scores = np.linspace(0.0, 1.0, 20)

    with pytest.raises(ValueError, match="single-class"):
        fit_isotonic(y, scores)


def test_reliability_curve_renders_without_error_on_balanced_input() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=200)
    scores = rng.uniform(size=200)

    figure = reliability_curve_figure(y, scores, n_bins=5)

    assert figure.axes[0].get_title() == "Reliability curve"
    assert figure.axes[0].get_xlabel() == "Mean predicted probability"


def test_reliability_curve_handles_single_class_gracefully() -> None:
    y = np.zeros(50, dtype=int)
    scores = np.linspace(0.0, 1.0, 50)

    figure = reliability_curve_figure(y, scores)

    assert figure.axes
