from typing import Any

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST

from fraud import registry


class _Model:
    def __init__(self, aliases: dict[str, str]) -> None:
        self.aliases = aliases


class _Version:
    def __init__(self, run_id: str | None) -> None:
        self.run_id = run_id


class _FakeClient:
    def __init__(
        self,
        *,
        model: Any = None,
        model_error: Exception | None = None,
        version: Any = None,
    ) -> None:
        self._model = model
        self._model_error = model_error
        self._version = version

    def get_registered_model(self, _name: str) -> Any:
        if self._model_error is not None:
            raise self._model_error
        return self._model

    def get_model_version(self, _name: str, _version: str) -> Any:
        return self._version


def _patch(monkeypatch: pytest.MonkeyPatch, client: _FakeClient) -> None:
    monkeypatch.setattr(registry, "MlflowClient", lambda: client)


def test_get_alias_version_returns_none_for_unregistered_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = MlflowException("no such model", error_code=RESOURCE_DOES_NOT_EXIST)
    _patch(monkeypatch, _FakeClient(model_error=error))

    assert registry.get_alias_version("argus", "champion") is None


def test_get_alias_version_reraises_other_mlflow_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    error = MlflowException("backend down", error_code=INTERNAL_ERROR)
    _patch(monkeypatch, _FakeClient(model_error=error))

    with pytest.raises(MlflowException):
        registry.get_alias_version("argus", "champion")


def test_get_alias_version_reads_the_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _FakeClient(model=_Model({"champion": "3"})))

    assert registry.get_alias_version("argus", "champion") == 3


def test_get_alias_version_returns_none_when_alias_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _FakeClient(model=_Model({})))

    assert registry.get_alias_version("argus", "champion") is None


def test_get_version_run_id_raises_when_run_id_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _FakeClient(version=_Version(None)))

    with pytest.raises(RuntimeError, match="no source run id"):
        registry.get_version_run_id("argus", 1)


def test_get_version_run_id_returns_the_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _FakeClient(version=_Version("run-9")))

    assert registry.get_version_run_id("argus", 1) == "run-9"
