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
    assert set(lineage.as_dict()) == {"git_sha", "dvc_lock_hash", "env_lock_hash"}
