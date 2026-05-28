import pathlib

import numpy as np

from fraud.common import collect_lineage, configure_logging, get_logger, set_seed


def test_set_seed_is_reproducible() -> None:
    set_seed(123)
    first = np.random.rand(5)
    set_seed(123)
    second = np.random.rand(5)
    assert np.array_equal(first, second)


def test_configure_logging_then_log() -> None:
    configure_logging(level="INFO", json_logs=True)
    logger = get_logger("test")
    logger.info("hello", key="value")


def test_collect_lineage_hashes_lockfile() -> None:
    lineage = collect_lineage()
    assert lineage.env_lock_hash is not None
    assert set(lineage.as_dict()) == {
        "git_sha",
        "git_dirty",
        "python_version",
        "dvc_lock_hash",
        "env_lock_hash",
    }


def test_lineage_to_mlflow_tags_are_all_strings() -> None:
    tags = collect_lineage().to_mlflow_tags()
    assert tags["python_version"].count(".") == 2
    assert all(isinstance(value, str) and value for value in tags.values())


def test_lineage_to_mlflow_tags_handles_missing_values(tmp_path: pathlib.Path) -> None:
    lineage = collect_lineage(tmp_path)  # no git, no dvc.lock, no uv.lock here
    tags = lineage.to_mlflow_tags()
    assert tags["git_sha"] == "unknown"
    assert tags["dvc_lock_hash"] == "unknown"
    assert tags["env_lock_hash"] == "unknown"
    assert tags["git_dirty"] == "unknown"
