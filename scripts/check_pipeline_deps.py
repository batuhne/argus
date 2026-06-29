"""Fail if a DVC stage under-tracks the first-party modules it imports."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _module_to_path(module: str) -> Path | None:
    rel = module.replace(".", "/")
    candidate = SRC / f"{rel}.py"
    if candidate.exists():
        return candidate
    package = SRC / rel / "__init__.py"
    if package.exists():
        return package
    return None


def _first_party_imports(path: Path, package: str) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names if alias.name.startswith("fraud"))
        elif isinstance(node, ast.ImportFrom):
            module = package if node.level else (node.module or "")
            if node.level and node.module:
                module = f"{package}.{node.module}"
            if module.startswith("fraud"):
                found.add(module)
                found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


def _resolve(module: str) -> tuple[str, Path] | None:
    path = _module_to_path(module)
    if path is not None:
        return module, path
    if "." in module:
        parent = module.rsplit(".", 1)[0]
        path = _module_to_path(parent)
        if path is not None:
            return parent, path
    return None


def _closure(entry: str) -> set[Path]:
    seen: set[str] = set()
    files: set[Path] = set()
    stack = [entry]
    while stack:
        resolved = _resolve(stack.pop())
        if resolved is None:
            continue
        module, path = resolved
        if module in seen:
            continue
        seen.add(module)
        files.add(path)
        package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
        stack.extend(_first_party_imports(path, package))
    return files


def _entrypoint(cmd: str) -> str | None:
    parts = cmd.split()
    if "-m" in parts:
        index = parts.index("-m")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def _declared_src_deps(deps: object) -> list[Path]:
    out: list[Path] = []
    if not isinstance(deps, list):
        return out
    for dep in deps:
        raw = dep.get("path") if isinstance(dep, dict) else dep
        if isinstance(raw, str):
            path = (ROOT / raw).resolve()
            if path == SRC or SRC in path.parents:
                out.append(path)
    return out


def _covered(target: Path, declared: list[Path]) -> bool:
    return any(dep == target or (dep.is_dir() and dep in target.parents) for dep in declared)


def _is_inert_init(path: Path) -> bool:
    """An __init__.py with no code (empty or docstring-only) cannot affect any output."""
    if path.name != "__init__.py":
        return False
    body = ast.parse(path.read_text()).body
    if not body:
        return True
    return (
        len(body) == 1 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
    )


def main() -> int:
    config = yaml.safe_load((ROOT / "dvc.yaml").read_text())
    stages = config.get("stages", {})
    problems: list[str] = []
    for name, spec in stages.items():
        if not isinstance(spec, dict):
            continue
        entry = _entrypoint(str(spec.get("cmd", "")))
        if entry is None:
            continue
        declared = _declared_src_deps(spec.get("deps"))
        for source in sorted(_closure(entry)):
            if _is_inert_init(source) or _covered(source, declared):
                continue
            problems.append(f"  {name}: imports {source.relative_to(ROOT)} but no dep covers it")
    if problems:
        print("DVC stage deps under-track the import closure:")
        print("\n".join(problems))
        return 1
    print(f"DVC stage deps cover the import closure for {len(stages)} stages.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
