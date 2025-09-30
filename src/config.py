"""Experiment configuration models defined in microseconds and megahertz."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping

__all__ = [
    "BaselineSpec",
    "PenaltyConfig",
    "ExperimentConfig",
    "override_from_dict",
]


@dataclass(slots=True)
class BaselineSpec:
    """Identify deterministic baseline parameters used to seed GRAPE runs."""

    name: str = "default"
    params: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name}
        if self.params is not None:
            payload["params"] = dict(self.params)
        else:
            payload["params"] = None
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "BaselineSpec":
        if not payload:
            return cls()
        name = str(payload.get("name", "default"))
        raw_params = payload.get("params")
        params = None if raw_params is None else dict(raw_params)
        return cls(name=name, params=params)


@dataclass(slots=True)
class PenaltyConfig:
    """Weights for optimizer penalties in MHz^2*us units."""

    power_weight: float = 0.0
    neg_weight: float = 0.0
    neg_kappa: float = 10.0

    def to_dict(self) -> dict[str, float]:
        return {
            "power_weight": float(self.power_weight),
            "neg_weight": float(self.neg_weight),
            "neg_kappa": float(self.neg_kappa),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "PenaltyConfig":
        if not payload:
            return cls()
        return cls(
            power_weight=float(payload.get("power_weight", 0.0)),
            neg_weight=float(payload.get("neg_weight", 0.0)),
            neg_kappa=float(payload.get("neg_kappa", 10.0)),
        )


@dataclass(slots=True)
class ExperimentConfig:
    """Container for experiment inputs expressed in microseconds and megahertz."""

    baseline: BaselineSpec = field(default_factory=BaselineSpec)
    run_name: str | None = None
    artifacts_root: Path | None = None
    random_seed: int | None = None
    optimizer_options: dict[str, Any] = field(default_factory=dict)
    penalties: PenaltyConfig = field(default_factory=PenaltyConfig)
    metadata: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentConfig":
        baseline = BaselineSpec.from_dict(payload.get("baseline"))
        penalties = PenaltyConfig.from_dict(payload.get("penalties"))
        metadata = dict(payload.get("metadata", {}))
        return cls(
            baseline=baseline,
            run_name=payload.get("run_name"),
            artifacts_root=Path(payload["artifacts_root"]) if payload.get("artifacts_root") else None,
            random_seed=payload.get("random_seed"),
            optimizer_options=dict(payload.get("optimizer_options", {})),
            penalties=penalties,
            metadata=metadata,
            notes=payload.get("notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "baseline": self.baseline.to_dict(),
            "run_name": self.run_name,
            "artifacts_root": str(self.artifacts_root) if self.artifacts_root is not None else None,
            "random_seed": self.random_seed,
            "optimizer_options": dict(self.optimizer_options),
            "penalties": self.penalties.to_dict(),
            "metadata": dict(self.metadata),
            "notes": self.notes,
        }
        return data

    def with_updates(self, overrides: Mapping[str, Any]) -> "ExperimentConfig":
        payload: MutableMapping[str, Any] = self.to_dict()
        payload.update(overrides)
        return ExperimentConfig.from_dict(payload)

    def slug(self) -> str:
        base = self.baseline.name if self.baseline.name else "run"
        return (self.run_name or base).replace(" ", "-").lower()


def _coerce_to_config(config: ExperimentConfig | Mapping[str, Any]) -> ExperimentConfig:
    """Return an ExperimentConfig from either an existing config or mapping."""

    if isinstance(config, ExperimentConfig):
        return config
    if isinstance(config, Mapping):
        return ExperimentConfig.from_dict(config)
    raise TypeError("Expected ExperimentConfig or mapping overrides.")


def override_from_dict(
    config: ExperimentConfig | Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> ExperimentConfig:
    """Create a new ExperimentConfig with fields updated from ``overrides``."""

    base = _coerce_to_config(config)
    return base.with_updates(overrides)
