from pathlib import Path

import numpy as np
import pandas as pd

from fraud.transforms import feature_logic as fl

UTC_BASE = pd.Timestamp("2017-12-01", tz="UTC")


def _events(seconds: list[int]) -> pd.Series:
    return pd.Series(UTC_BASE + pd.to_timedelta(seconds, unit="s"))


def test_to_event_timestamp_offsets_from_reference() -> None:
    out = fl.to_event_timestamp(pd.Series([0, 86400]))

    assert out.iloc[0] == fl.REFERENCE_DATETIME
    assert out.iloc[1] == fl.REFERENCE_DATETIME + pd.Timedelta(days=1)


def test_make_card_id_is_stable_and_handles_missing_parts() -> None:
    frame = pd.DataFrame(
        {
            "card1": [1, 1],
            "card2": [2.0, np.nan],
            "card3": [3.0, 3.0],
            "card5": [5.0, 5.0],
        }
    )

    card_id = fl.make_card_id(frame)

    assert card_id.tolist() == ["1_2_3_5", "1_na_3_5"]
    assert card_id.equals(fl.make_card_id(frame))


def test_card_velocity_uses_only_prior_transactions() -> None:
    # The first transaction sees no history; the second sees only the first; and
    # one more than 24h later sees an empty window again.
    frame = pd.DataFrame(
        {
            "card_id": ["A", "A", "A", "B"],
            "event_timestamp": _events([0, 3600, 100_000, 50]),
            "TransactionAmt": [10.0, 20.0, 5.0, 7.0],
        }
    )

    out = fl.compute_card_velocity(frame)
    out["seconds"] = (out["event_timestamp"] - UTC_BASE).dt.total_seconds().astype(int)

    def value(card_id: str, seconds: int, column: str) -> float:
        row = out.loc[(out["card_id"] == card_id) & (out["seconds"] == seconds), column]
        return float(row.iloc[0])

    assert value("A", 0, "card_txn_count_24h") == 0.0
    assert value("A", 0, "card_amt_sum_24h") == 0.0
    assert value("A", 0, "seconds_since_prev_txn") == fl.NO_PRIOR_TXN

    assert value("A", 3600, "card_txn_count_24h") == 1.0
    assert value("A", 3600, "card_amt_sum_24h") == 10.0
    assert value("A", 3600, "card_amt_mean_24h") == 10.0
    assert value("A", 3600, "card_amt_max_24h") == 10.0
    assert value("A", 3600, "seconds_since_prev_txn") == 3600.0

    assert value("A", 100_000, "card_txn_count_24h") == 0.0
    assert value("A", 100_000, "card_amt_sum_24h") == 0.0
    assert value("A", 100_000, "seconds_since_prev_txn") == 96_400.0

    assert value("B", 50, "card_txn_count_24h") == 0.0
    assert value("B", 50, "seconds_since_prev_txn") == fl.NO_PRIOR_TXN


def test_amount_log_matches_log1p() -> None:
    amount = pd.Series([0.0, 9.0, 99.0])

    np.testing.assert_allclose(fl.amount_log(amount).to_numpy(), np.log1p(amount.to_numpy()))


def test_amount_to_mean_ratio_is_neutral_without_history() -> None:
    amount = pd.Series([20.0, 10.0, 10.0])
    mean = pd.Series([10.0, 0.0, np.nan])

    ratio = fl.amount_to_mean_ratio(amount, mean)

    assert ratio.tolist() == [2.0, 1.0, 1.0]


def test_load_v_selected_returns_empty_when_file_absent(tmp_path: Path) -> None:
    assert fl.load_v_selected(tmp_path / "missing.json") == ()


def test_load_v_selected_reads_frozen_list(tmp_path: Path) -> None:
    path = tmp_path / "v_selected.json"
    path.write_text('["V1", "V2"]')

    assert fl.load_v_selected(path) == ("V1", "V2")


def test_coerce_numeric_casts_to_float32_and_keeps_nan() -> None:
    frame = pd.DataFrame({"C1": [1, 2, 3], "D1": [1.5, None, 3.5]})

    out = fl.coerce_numeric(frame, ["C1", "D1"])

    assert out["C1"].dtype == "float32"
    assert out["D1"].dtype == "float32"
    assert out["C1"].tolist() == [1.0, 2.0, 3.0]
    assert pd.isna(out["D1"].iloc[1])


def test_coerce_numeric_turns_unparseable_into_nan() -> None:
    out = fl.coerce_numeric(pd.DataFrame({"C1": ["x", "2.0"]}), ["C1"])

    assert pd.isna(out["C1"].iloc[0])
    assert out["C1"].iloc[1] == 2.0


def test_coerce_numeric_does_not_mutate_input() -> None:
    frame = pd.DataFrame({"C1": ["x"]})

    fl.coerce_numeric(frame, ["C1"])

    assert frame["C1"].iloc[0] == "x"


def test_amount_to_mean_ratio_handles_null_mean_from_unknown_card() -> None:
    # An unseen card has no online velocity, so Feast returns a null mean as a
    # Python None on an object column; serving must score it, not 500.
    amount = pd.Series([59.0])
    mean = pd.Series([None], dtype="object")

    ratio = fl.amount_to_mean_ratio(amount, mean)

    assert ratio.tolist() == [1.0]
