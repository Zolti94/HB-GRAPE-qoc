"""Utilities to activate the repository context inside notebooks."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

DEFAULT_MARKERS: Sequence[str] = ("pyproject.toml", "requirements.txt", "src")


def _iter_candidates(start: Path) -> Iterable[Path]:
    current = start.resolve()
    yield current
    yield from current.parents


def _has_marker(path: Path, markers: Sequence[str]) -> bool:
    for marker in markers:
        if (path / marker).exists():
            return True
    return False


def find_repo_root(start: Path | None = None, *, markers: Sequence[str] = DEFAULT_MARKERS) -> Path:
    """Return the repository root by scanning upwards from ``start``."""

    start_path = Path(start or Path.cwd()).resolve()
    for candidate in _iter_candidates(start_path):
        if _has_marker(candidate, markers):
            return candidate
    raise RuntimeError(f"Unable to locate repository root from {start_path}")


def activate_repository(
    context: Mapping[str, object] | None = None,
    *,
    change_cwd: bool = True,
    add_sys_path: bool = True,
    markers: Sequence[str] = DEFAULT_MARKERS,
) -> Path:
    """Ensure notebooks run with the repository root on ``sys.path``.

    Parameters
    ----------
    context:
        Mapping of globals (for example ``globals()``) used to resolve ``__file__``.
    change_cwd:
        When ``True`` (default) change the current working directory to the root.
    add_sys_path:
        When ``True`` (default) prepend the repository root to ``sys.path``.
    markers:
        Optional override for the filesystem markers that delimit the root.

    Returns
    -------
    pathlib.Path
        The detected repository root.
    """

    if context is not None and "__file__" in context:
        start = Path(str(context["__file__"])).resolve().parent
    else:
        start = Path.cwd()

    repo_root = find_repo_root(start, markers=markers)

    if add_sys_path:
        root_str = str(repo_root)
        if root_str not in map(str, sys.path):
            sys.path.insert(0, root_str)

    if change_cwd and Path.cwd() != repo_root:
        os.chdir(repo_root)

    return repo_root


__all__ = ["activate_repository", "find_repo_root"]
