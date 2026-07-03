from pathlib import Path

import pandas as pd

from fraud.features.build_offline import build_offline_features


def test_build_offline_features_writes_atomically(tmp_path: Path) -> None:
    input_path = tmp_path / "clean.parquet"
    output_path = tmp_path / "card_features.parquet"
    pd.DataFrame(
        {
            "card1": [1, 1, 2],
            "card2": [10, 10, 20],
            "card3": [100, 100, 200],
            "card5": [1000, 1000, 2000],
            "TransactionDT": [86400, 90000, 86400],
            "TransactionAmt": [10.0, 20.0, 30.0],
        }
    ).to_parquet(input_path)

    build_offline_features(input_path=input_path, output_path=output_path)

    assert output_path.exists()
    assert not output_path.with_name(output_path.name + ".tmp").exists()
    written = pd.read_parquet(output_path)
    assert "card_id" in written.columns
    assert len(written) == 3
