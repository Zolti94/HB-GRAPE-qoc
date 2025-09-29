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
    """Return the artifact root directory for ``config``.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment description containing an optional custom artifact path.

    Returns
    -------
    pathlib.Path
        Directory where run outputs should be written; defaults to ``DEFAULT_ROOT``.
    """

    return config.artifacts_root if config.artifacts_root is not None else DEFAULT_ROOT


def _timestamp_string(timestamp: datetime | None = None) -> str:
    """Format ``timestamp`` as ``YYYYMMDD-HHMMSS``, defaulting to ``datetime.utcnow``."""

    stamp = timestamp or datetime.utcnow()
    return stamp.strftime("%Y%m%d-%H%M%S")


def format_run_name(config: ExperimentConfig, method: str, *, timestamp: datetime | None = None) -> str:
    """Compose a filesystem-friendly run name.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration providing a slug derived from the baseline/run name.
    method : str
        Optimizer identifier (e.g., ``"adam"``) prepended to the slug.
    timestamp : datetime, optional
        Timestamp appended for uniqueness; defaults to ``datetime.utcnow``.

    Returns
    -------
    str
        Hyphen-separated identifier ``method-baseline-YYYYMMDD-HHMMSS``.
    """

    base = config.slug()
    components = [method, base, _timestamp_string(timestamp)]
    return "-".join(filter(None, components))


@dataclass(slots=True)
class ArtifactPaths:
    """Container describing all filesystem outputs produced by ``run_experiment``.

    Attributes
    ----------
    run_dir, config_json, metrics_json, history_npz, pulses_npz, figures_dir, log_path : Path
        Concrete paths within the artifact root for the run directory, serialized
        configuration/metrics, NumPy history and pulse archives, the figures folder,
        and the status log file.
    """

    run_dir: Path
    config_json: Path
    metrics_json: Path
    history_npz: Path
    pulses_npz: Path
    figures_dir: Path
    log_path: Path

    def as_dict(self) -> dict[str, Any]:
        """Return the path mapping as JSON-ready strings."""

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
    """Create the run directory tree and return the resolved paths.

    Parameters
    ----------
    run_name : str
        Unique identifier for the run directory.
    root : Path, optional
        Base artifacts directory; defaults to ``DEFAULT_ROOT`` when omitted.
    exist_ok : bool, optional
        Whether to reuse an existing directory instead of raising ``FileExistsError``.

    Returns
    -------
    ArtifactPaths
        Named paths pointing to all files generated during the experiment.
    """

    run_root = root or DEFAULT_ROOT
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=exist_ok)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    # Pre-create figures directory so downstream plotting utilities can assume existence.
    return ArtifactPaths(
        run_dir=run_dir,
        config_json=run_dir / "config.json",
        metrics_json=run_dir / "metrics.json",
        history_npz=run_dir / "history.npz",
        pulses_npz=run_dir / "pulses.npz",
        figures_dir=figures_dir,
        log_path=run_dir / "logs.txt",
    )
