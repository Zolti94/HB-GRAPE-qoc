"""Metric extraction utilities shared by notebooks and CLI tools."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import numpy as np

from ..result import Result
from ..config import ExperimentConfig


def _to_array(series: Any, *, dtype: Any = float) -> np.ndarray | None:
    if series is None:
        return None
    arr = np.asarray(series, dtype=dtype)
    return arr if arr.size else None


@dataclass(slots=True)
class HistorySeries:
    """Subset of optimizer traces normalised as NumPy arrays."""

    total: np.ndarray | None = None
    terminal: np.ndarray | None = None
    power_penalty: np.ndarray | None = None
    neg_penalty: np.ndarray | None = None
    grad_norm: np.ndarray | None = None
    step_norm: np.ndarray | None = None
    lr: np.ndarray | None = None
    calls_per_iter: np.ndarray | None = None
    iterations: np.ndarray | None = None

    def length(self) -> int:
        for series in (
            self.total,
            self.terminal,
            self.grad_norm,
            self.step_norm,
            self.iterations,
        ):
            if series is not None:
                return int(series.size)
        return 0


@dataclass(slots=True)
class PulseSummary:
    """Waveform snapshot aggregated from optimizer output."""

    t_us: np.ndarray
    omega: np.ndarray
    delta: np.ndarray | None
    omega_base: np.ndarray | None
    delta_base: np.ndarray | None


@dataclass(slots=True)
class ResultSummary:
    """Compact view of experiment outputs used by downstream analysis."""

    run_name: str
    artifacts_dir: Path
    history: HistorySeries
    pulses: PulseSummary
    metrics: dict[str, float]
    metadata: dict[str, Any]
    oracle_calls: int
    runtime_s: float
    iterations: int
    grad_norm_final: float | None
    step_norm_final: float | None
    max_abs_omega: float
    area_omega_over_pi: float
    negativity_fraction: float
    status: str


def collect_history_series(history: Mapping[str, Any], *, keep_lr: bool = True) -> HistorySeries:
    """Normalise history payloads emitted by optimizers."""

    mapping = dict(history)
    iter_series = mapping.get("iter")
    if iter_series is None and "total" in mapping:
        iter_series = np.arange(1, len(mapping["total"]) + 1, dtype=np.int64)
    return HistorySeries(
        total=_to_array(mapping.get("total")),
        terminal=_to_array(mapping.get("terminal")),
        power_penalty=_to_array(mapping.get("power_penalty")),
        neg_penalty=_to_array(mapping.get("neg_penalty")),
        grad_norm=_to_array(mapping.get("grad_norm")),
        step_norm=_to_array(mapping.get("step_norm")),
        lr=_to_array(mapping.get("lr")) if keep_lr else None,
        calls_per_iter=_to_array(mapping.get("calls_per_iter"), dtype=np.int64),
        iterations=_to_array(iter_series, dtype=np.int64),
    )


def _pulse_summary_from_result(result: Result) -> PulseSummary:
    pulses = result.pulses or {}
    omega = np.asarray(pulses.get("omega"), dtype=np.float64)
    t_us = np.asarray(pulses.get("t_us"), dtype=np.float64)
    delta_raw = pulses.get("delta")
    delta = None if delta_raw is None else np.asarray(delta_raw, dtype=np.float64)
    omega_base_raw = pulses.get("omega_base")
    omega_base = (
        None if omega_base_raw is None else np.asarray(omega_base_raw, dtype=np.float64)
    )
    delta_base_raw = pulses.get("delta_base")
    delta_base = (
        None if delta_base_raw is None else np.asarray(delta_base_raw, dtype=np.float64)
    )
    return PulseSummary(
        t_us=t_us,
        omega=omega,
        delta=delta,
        omega_base=omega_base,
        delta_base=delta_base,
    )


def summarize_result(result: Result) -> ResultSummary:
    """Derive scalar metrics and waveform summaries from a :class:`Result`."""

    history = collect_history_series(result.history)
    metrics = dict(result.final_metrics)

    runtime_s = float(metrics.get("runtime_s", np.nan))
    calls_series = history.calls_per_iter
    oracle_calls = int(calls_series.sum()) if calls_series is not None else 0

    grad_final = (
        float(history.grad_norm[-1]) if history.grad_norm is not None and history.grad_norm.size else None
    )
    step_final = (
        float(history.step_norm[-1]) if history.step_norm is not None and history.step_norm.size else None
    )
    iterations = history.length()

    pulses = _pulse_summary_from_result(result)
    omega = pulses.omega
    t_us = pulses.t_us
    if omega.size == 0 or t_us.size == 0:
        max_abs_omega = float("nan")
        area_pi = float("nan")
        neg_frac = float("nan")
    else:
        max_abs_omega = float(np.max(np.abs(omega)))
        area_pi = float(np.trapezoid(np.abs(omega), t_us) / np.pi)
        neg_frac = float(np.mean(omega < 0.0))

    status_payload = result.optimizer_state or {}
    status = str(status_payload.get("status", "unknown"))

    return ResultSummary(
        run_name=result.run_name,
        artifacts_dir=Path(result.artifacts_dir),
        history=history,
        pulses=pulses,
        metrics=metrics,
        metadata=dict(getattr(result.config, "metadata", {}) or {}),
        oracle_calls=oracle_calls,
        runtime_s=runtime_s,
        iterations=iterations,
        grad_norm_final=grad_final,
        step_norm_final=step_final,
        max_abs_omega=max_abs_omega,
        area_omega_over_pi=area_pi,
        negativity_fraction=neg_frac,
        status=status,
    )


def load_result_summary(run_dir: Path) -> ResultSummary:
    """Load persisted artifacts and return a :class:`ResultSummary`."""

    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    history_path = run_dir / "history.npz"
    pulses_path = run_dir / "pulses.npz"
    config_path = run_dir / "config.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {run_dir}")
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history.npz in {run_dir}")
    if not pulses_path.exists():
        raise FileNotFoundError(f"Missing pulses.npz in {run_dir}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    history_npz = np.load(history_path)
    pulses_npz = np.load(pulses_path)

    history: MutableMapping[str, np.ndarray] = {}
    for key in history_npz.files:
        history[key] = np.asarray(history_npz[key])

    pulses: MutableMapping[str, np.ndarray] = {}
    for key in pulses_npz.files:
        pulses[key] = np.asarray(pulses_npz[key])

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")

    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = ExperimentConfig.from_dict(config_payload)
    run_name = config_payload.get("run_name") or run_dir.name

    result = Result(
        run_name=run_name,
        artifacts_dir=run_dir,
        config=config,
        history=history,
        final_metrics=metrics,
        pulses=pulses,
    )
    result.optimizer_state = {"status": metrics.get("status", "unknown")}
    return summarize_result(result)
