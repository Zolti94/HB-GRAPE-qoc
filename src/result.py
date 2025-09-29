"""Result container for optimization runs in microseconds and megahertz."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import ExperimentConfig

__all__ = ["Result", "ResultPayload"]

ResultPayload = Mapping[str, Any]


def _json_ready(value: Any) -> Any:
    """Convert nested experiment outputs to JSON-serializable objects.

    Parameters
    ----------
    value : Any
        Payload captured during optimization (may include NumPy arrays, paths, or datetimes).

    Returns
    -------
    Any
        Plain Python equivalent suitable for ``json.dumps``.
    """

    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


@dataclass(slots=True)
class Result:
    """Structured record capturing the outcome of a single experiment run.

    Parameters
    ----------
    run_name : str
        Identifier assigned to the run directory.
    artifacts_dir : pathlib.Path
        Filesystem location containing saved artifacts.
    config : ExperimentConfig
        Configuration used to launch the experiment.
    history : dict[str, numpy.ndarray], optional
        Time-series traces gathered during optimization.
    final_metrics : dict[str, float], optional
        Scalar metrics reported at completion.
    pulses : dict[str, numpy.ndarray], optional
        Control waveforms and metadata emitted by the optimizer.
    created_at : datetime, optional
        Timestamp capturing when the result object was created.
    optimizer_state : dict[str, Any], optional
        Optimizer-specific diagnostic information.
    """

    run_name: str
    artifacts_dir: Path
    config: ExperimentConfig
    history: dict[str, np.ndarray] = field(default_factory=dict)
    final_metrics: dict[str, float] = field(default_factory=dict)
    pulses: dict[str, np.ndarray] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    optimizer_state: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to primitives for persistence or logging."""

        return {
            "run_name": self.run_name,
            "artifacts_dir": str(self.artifacts_dir),
            "config": self.config.to_dict(),
            "history": {k: _json_ready(v) for k, v in self.history.items()},
            "final_metrics": {k: float(v) for k, v in self.final_metrics.items()},
            "pulses": {k: _json_ready(v) for k, v in self.pulses.items()},
            "created_at": self.created_at.isoformat(),
            "optimizer_state": _json_ready(self.optimizer_state) if self.optimizer_state is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Result":
        """Recreate a :class:`Result` instance from serialized payloads."""

        config = ExperimentConfig.from_dict(payload["config"])
        run_name = str(payload["run_name"])
        artifacts_dir = Path(payload["artifacts_dir"])
        history = {str(k): np.asarray(v) for k, v in payload.get("history", {}).items()}
        final_metrics = {str(k): float(v) for k, v in payload.get("final_metrics", {}).items()}
        pulses = {str(k): np.asarray(v) for k, v in payload.get("pulses", {}).items()}
        created_at_raw = payload.get("created_at")
        created_at = datetime.fromisoformat(created_at_raw) if created_at_raw else datetime.utcnow()
        optimizer_state = payload.get("optimizer_state")
        return cls(
            run_name=run_name,
            artifacts_dir=artifacts_dir,
            config=config,
            history=history,
            final_metrics=final_metrics,
            pulses=pulses,
            created_at=created_at,
            optimizer_state=optimizer_state,
        )

    def summary(self) -> dict[str, Any]:
        """Return a compact overview consisting of run name, cost, and iterations."""

        cost = float(self.final_metrics.get("cost", np.nan))
        # All history series share a length; use any entry to infer iteration count.
        iterations = int(len(next(iter(self.history.values()))) if self.history else 0)
        return {"run_name": self.run_name, "cost": cost, "iterations": iterations}
