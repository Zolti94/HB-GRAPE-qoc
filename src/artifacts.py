"""Artifact path helpers keeping microsecond/megahertz semantics."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ExperimentConfig

__all__ = [
    "ArtifactPaths",
    "default_root",
    "format_run_name",
    "prepare_run_directory",
]

DEFAULT_ROOT = Path("artifacts")


def default_root(config: ExperimentConfig) -> Path:
    return config.artifacts_root if config.artifacts_root is not None else DEFAULT_ROOT


def _timestamp_string(timestamp: datetime | None = None) -> str:
    stamp = timestamp or datetime.utcnow()
    return stamp.strftime("%Y%m%d-%H%M%S")


def format_run_name(config: ExperimentConfig, method: str, *, timestamp: datetime | None = None) -> str:
    base = config.slug()
    components = [method, base, _timestamp_string(timestamp)]
    return "-".join(filter(None, components))


@dataclass(slots=True)
class ArtifactPaths:
    run_dir: Path
    config_json: Path
    metrics_json: Path
    history_npz: Path
    pulses_npz: Path
    figures_dir: Path
    log_path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "config_json": str(self.config_json),
            "metrics_json": str(self.metrics_json),
            "history_npz": str(self.history_npz),
            "pulses_npz": str(self.pulses_npz),
            "figures_dir": str(self.figures_dir),
            "log_path": str(self.log_path),
        }


def prepare_run_directory(
    run_name: str,
    *,
    root: Path | None = None,
    exist_ok: bool = False,
) -> ArtifactPaths:
    run_root = root or DEFAULT_ROOT
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=exist_ok)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return ArtifactPaths(
        run_dir=run_dir,
        config_json=run_dir / "config.json",
        metrics_json=run_dir / "metrics.json",
        history_npz=run_dir / "history.npz",
        pulses_npz=run_dir / "pulses.npz",
        figures_dir=figures_dir,
        log_path=run_dir / "logs.txt",
    )
