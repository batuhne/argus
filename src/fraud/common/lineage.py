"""Collect lineage tags (git SHA and dirty flag, Python version, file hashes) for a run."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Lineage:
    """Provenance for a run: which code, which data, which locked environment."""

    git_sha: str | None
    git_dirty: bool | None
    python_version: str
    dvc_lock_hash: str | None
    env_lock_hash: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_mlflow_tags(self) -> dict[str, str]:
        return {
            "git_sha": self.git_sha or "unknown",
            "git_dirty": "unknown" if self.git_dirty is None else str(self.git_dirty).lower(),
            "python_version": self.python_version,
            "dvc_lock_hash": self.dvc_lock_hash or "unknown",
            "env_lock_hash": self.env_lock_hash or "unknown",
        }


def collect_lineage(root: Path | None = None) -> Lineage:
    base = root or Path.cwd()
    return Lineage(
        git_sha=_git_sha(base),
        git_dirty=_git_dirty(base),
        python_version=_python_version(),
        dvc_lock_hash=_hash_file(base / "dvc.lock"),
        env_lock_hash=_hash_file(base / "uv.lock"),
    )


def _git_sha(root: Path) -> str | None:
    return _run(["git", "rev-parse", "HEAD"], cwd=root)


def _git_dirty(root: Path) -> bool | None:
    porcelain = _run(["git", "status", "--porcelain"], cwd=root)
    return None if porcelain is None else bool(porcelain)


def _python_version() -> str:
    return ".".join(str(part) for part in sys.version_info[:3])


def _run(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return sha256_file(path)
