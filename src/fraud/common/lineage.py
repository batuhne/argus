from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Lineage:
    """Provenance for a run: which code, which data, which locked environment."""

    git_sha: str | None
    dvc_lock_hash: str | None
    env_lock_hash: str | None

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


def collect_lineage(root: Path | None = None) -> Lineage:
    base = root or Path.cwd()
    return Lineage(
        git_sha=_git_sha(base),
        dvc_lock_hash=_hash_file(base / "dvc.lock"),
        env_lock_hash=_hash_file(base / "uv.lock"),
    )


def _git_sha(root: Path) -> str | None:
    return _run(["git", "rev-parse", "HEAD"], cwd=root)


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


def _hash_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()
