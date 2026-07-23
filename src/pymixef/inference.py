"""Restartable deterministic bootstrap workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._serialization import read_json, write_json
from .diagnostics import DiagnosticTable
from .random import RandomStreamManager
from .results import FitResult


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Parameter draws, failure accounting, and interval calculations."""

    draws: DiagnosticTable
    failures: tuple[Mapping[str, Any], ...]
    seed: int
    resampling: str

    @property
    def successful_replicates(self) -> int:
        return len(self.draws)

    @property
    def failed_replicates(self) -> int:
        return len(self.failures)

    def intervals(self, level: float = 0.95, *, method: str = "percentile") -> DiagnosticTable:
        if not 0 < level < 1:
            raise ValueError("level must lie strictly between zero and one.")
        if method != "percentile":
            raise NotImplementedError(
                "The portable bootstrap result implements percentile intervals; "
                "BCa requires influence values from a compatible backend."
            )
        alpha = (1.0 - level) / 2.0
        names: list[str] = []
        lower: list[float] = []
        median: list[float] = []
        upper: list[float] = []
        for name, values in self.draws.columns.items():
            if name == "replicate":
                continue
            numeric = np.asarray(values, dtype=float)
            names.append(name)
            lower.append(float(np.quantile(numeric, alpha)))
            median.append(float(np.median(numeric)))
            upper.append(float(np.quantile(numeric, 1.0 - alpha)))
        return DiagnosticTable(
            "bootstrap_intervals",
            {
                "parameter": np.asarray(names),
                "lower": np.asarray(lower),
                "median": np.asarray(median),
                "upper": np.asarray(upper),
            },
            {
                "level": level,
                "method": method,
                "successful_replicates": self.successful_replicates,
                "failed_replicates": self.failed_replicates,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "draws": self.draws.to_dict(),
            "failures": [dict(item) for item in self.failures],
            "seed": self.seed,
            "resampling": self.resampling,
        }


def _columns(data: Any) -> dict[str, np.ndarray]:
    if hasattr(data, "to_dict"):
        try:
            data = data.to_dict(orient="list")
        except TypeError:
            data = data.to_dict()
    if not isinstance(data, Mapping):
        raise TypeError("Bootstrap data must be a column mapping or data-frame-like.")
    columns = {str(name): np.asarray(value) for name, value in data.items()}
    lengths = {len(value) for value in columns.values()}
    if len(lengths) != 1:
        raise ValueError("Bootstrap data columns must have equal lengths.")
    return columns


def _cluster_indices(
    cluster: np.ndarray, generator: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    levels: list[Any] = []
    lookup: set[Any] = set()
    for item in cluster.tolist():
        if item not in lookup:
            lookup.add(item)
            levels.append(item)
    selected = generator.choice(len(levels), size=len(levels), replace=True)
    pieces: list[np.ndarray] = []
    relabeled: list[np.ndarray] = []
    for new_cluster, choice in enumerate(selected):
        rows = np.flatnonzero(cluster == levels[int(choice)])
        pieces.append(rows)
        relabeled.append(np.full(rows.size, new_cluster, dtype=int))
    if not pieces:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    return np.concatenate(pieces), np.concatenate(relabeled)


def bootstrap(
    fit_function: Callable[[Mapping[str, np.ndarray]], FitResult],
    data: Any,
    *,
    n_replicates: int,
    seed: int,
    cluster: str | None = None,
    checkpoint: str | Path | None = None,
    resume: bool = True,
) -> BootstrapResult:
    """Run a nonparametric row or cluster bootstrap with restartable checkpoints."""

    if n_replicates < 1:
        raise ValueError("n_replicates must be positive.")
    columns = _columns(data)
    n_rows = len(next(iter(columns.values())))
    if cluster is not None and cluster not in columns:
        raise KeyError(f"Cluster column {cluster!r} is absent.")
    parameter_draws: dict[str, list[float]] = {}
    completed: list[int] = []
    failures: list[dict[str, Any]] = []
    checkpoint_path = None if checkpoint is None else Path(checkpoint)
    if checkpoint_path is not None and checkpoint_path.exists() and resume:
        saved = read_json(checkpoint_path)
        if int(saved["seed"]) != int(seed):
            raise ValueError("Checkpoint seed does not match the requested seed.")
        if int(saved["n_replicates"]) != n_replicates:
            raise ValueError("Checkpoint replicate count does not match.")
        completed = [int(item) for item in saved.get("completed", ())]
        parameter_draws = {
            name: [float(item) for item in values]
            for name, values in saved.get("parameter_draws", {}).items()
        }
        failures = [dict(item) for item in saved.get("failures", ())]
    streams = RandomStreamManager(seed, "pymixef-bootstrap")
    for replicate in range(n_replicates):
        if replicate in completed:
            continue
        generator = streams.generator("resample", replicate=replicate)
        if cluster is None:
            indices = generator.integers(0, n_rows, size=n_rows)
            sample = {name: values[indices] for name, values in columns.items()}
        else:
            indices, relabeled = _cluster_indices(columns[cluster], generator)
            sample = {name: values[indices] for name, values in columns.items()}
            sample[cluster] = relabeled
        try:
            fit = fit_function(sample)
            if fit.convergence.status == "failed":
                raise RuntimeError("fit returned failed convergence status")
            if not parameter_draws:
                parameter_draws = {name: [] for name in fit.parameters}
            if set(fit.parameters) != set(parameter_draws):
                raise RuntimeError("parameter set changed across bootstrap replicates")
            for name, value in fit.parameters.items():
                parameter_draws[name].append(float(value))
        except Exception as error:
            failures.append(
                {
                    "replicate": replicate,
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
        completed.append(replicate)
        if checkpoint_path is not None:
            write_json(
                checkpoint_path,
                {
                    "schema_version": "1.0.0",
                    "seed": seed,
                    "n_replicates": n_replicates,
                    "completed": completed,
                    "parameter_draws": parameter_draws,
                    "failures": failures,
                },
            )
    successful = len(next(iter(parameter_draws.values()))) if parameter_draws else 0
    return BootstrapResult(
        draws=DiagnosticTable(
            "bootstrap_parameter_draws",
            {
                "replicate": np.arange(successful),
                **{
                    name: np.asarray(values, dtype=float)
                    for name, values in parameter_draws.items()
                },
            },
            {
                "requested_replicates": n_replicates,
                "successful_replicates": successful,
                "failed_replicates": len(failures),
                "seed": seed,
                "cluster": cluster,
            },
        ),
        failures=tuple(failures),
        seed=seed,
        resampling="cluster" if cluster is not None else "row",
    )
