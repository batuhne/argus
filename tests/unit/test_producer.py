from fraud.ingestion.producer import _optional_float


def test_optional_float_passes_finite_values() -> None:
    assert _optional_float(3.5) == 3.5
    assert _optional_float(0) == 0.0


def test_optional_float_maps_blank_to_none() -> None:
    assert _optional_float(None) is None
    assert _optional_float(float("nan")) is None


def test_optional_float_maps_non_finite_to_none() -> None:
    assert _optional_float(float("inf")) is None
    assert _optional_float(float("-inf")) is None
