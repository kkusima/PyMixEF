"""Diagnostic calculations that return data before presentation."""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._serialization import read_json, write_json
from .data import adapt_data, is_missing


@dataclass(frozen=True, slots=True)
class DiagnosticTable:
    """A tidy, serializable numeric diagnostic table with calculation metadata."""

    name: str
    columns: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        converted = {str(name): np.asarray(values) for name, values in self.columns.items()}
        lengths = {len(values) for values in converted.values()}
        if len(lengths) > 1:
            raise ValueError(f"Diagnostic table {self.name!r} has unequal column lengths.")
        object.__setattr__(self, "columns", converted)

    def __len__(self) -> int:
        return len(next(iter(self.columns.values()))) if self.columns else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "columns": dict(self.columns),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DiagnosticTable:
        return cls(
            name=str(value["name"]),
            columns={name: np.asarray(column) for name, column in dict(value["columns"]).items()},
            metadata=dict(value.get("metadata", {})),
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        if destination.suffix.lower() == ".json":
            return write_json(destination, self.to_dict())
        if destination.suffix.lower() == ".csv":
            destination.parent.mkdir(parents=True, exist_ok=True)
            names = list(self.columns)
            with destination.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(names)
                writer.writerows(zip(*(self.columns[name] for name in names), strict=True))
            write_json(
                destination.with_suffix(".metadata.json"),
                {"name": self.name, "metadata": dict(self.metadata)},
            )
            return destination
        raise ValueError("Diagnostic tables can be saved as .json or .csv.")

    @classmethod
    def load(cls, path: str | Path) -> DiagnosticTable:
        source = Path(path)
        if source.suffix.lower() == ".json":
            return cls.from_dict(read_json(source))
        if source.suffix.lower() == ".csv":
            with source.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            metadata_path = source.with_suffix(".metadata.json")
            metadata = read_json(metadata_path) if metadata_path.exists() else {}
            columns: dict[str, np.ndarray] = {}
            if rows:
                for name in rows[0]:
                    raw = [row[name] for row in rows]
                    try:
                        columns[name] = np.asarray(raw, dtype=float)
                    except ValueError:
                        columns[name] = np.asarray(raw, dtype=str)
            return cls(
                name=str(metadata.get("name", source.stem)),
                columns=columns,
                metadata=dict(metadata.get("metadata", {})),
            )
        raise ValueError("Diagnostic tables can be loaded from .json or .csv.")


@dataclass(frozen=True, slots=True)
class GroupInfluenceResult:
    """Delete-whole-group refits with optional approximation comparisons."""

    table: DiagnosticTable
    failures: tuple[Mapping[str, Any], ...]
    group_column: str
    requested_groups: int

    @property
    def successful_groups(self) -> int:
        """Number of grouping levels with a completed full refit."""

        if not len(self.table):
            return 0
        return len(np.unique(self.table.columns["group_index"]))

    @property
    def failed_groups(self) -> int:
        """Number of grouping levels whose full refit failed."""

        return sum(item.get("phase") == "full-refit" for item in self.failures)

    def to_dict(self) -> dict[str, Any]:
        """Return the table, failures, and group-level accounting."""

        return {
            "table": self.table.to_dict(),
            "failures": [dict(item) for item in self.failures],
            "group_column": self.group_column,
            "requested_groups": self.requested_groups,
            "successful_groups": self.successful_groups,
            "failed_groups": self.failed_groups,
        }


def _group_levels(values: np.ndarray) -> tuple[Any, ...]:
    levels: list[Any] = []
    seen: set[Any] = set()
    for raw in values:
        value = raw.item() if isinstance(raw, np.generic) else raw
        if is_missing(value):
            raise ValueError("Grouping-safe influence analysis does not accept missing groups.")
        try:
            known = value in seen
        except TypeError as error:
            raise TypeError("Grouping values must be hashable scalars.") from error
        if not known:
            seen.add(value)
            levels.append(value)
    return tuple(levels)


def _fit_model_rank(fit: Any) -> int:
    engine_metrics = getattr(fit.convergence, "engine_metrics", {})
    extra = getattr(fit, "extra", {})
    value = engine_metrics.get("fixed_effect_rank")
    if value is None:
        value = extra.get("fixed_effect_rank")
    if value is None:
        raise RuntimeError(
            "The fit did not archive fixed_effect_rank required for influence safety."
        )
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise RuntimeError("The archived fixed_effect_rank must be a nonnegative integer.")
    return int(numeric)


def group_influence(
    fit_function: Callable[[Mapping[str, np.ndarray]], Any],
    data: Any,
    *,
    group: str,
    baseline: Any | None = None,
    approximation: Callable[[Any, Any], Mapping[str, float]] | None = None,
) -> GroupInfluenceResult:
    """Measure influence by deleting complete grouping levels and refitting.

    ``fit_function`` receives a column mapping and must return a fit-like object
    with ``parameters``, finite ``objective``, ``convergence``, and an archived
    ``fixed_effect_rank`` in convergence engine metrics or result extras;
    ``parameter_covariance`` is optional. If ``baseline`` is omitted, the
    callback first fits the full data. Every subsequent callback receives all
    rows except one complete grouping level—individual rows are never deleted
    in isolation.

    An optional ``approximation(group_value, baseline)`` callback can return
    approximate delete-group parameter estimates. The result then records the
    approximation error beside the full-refit change, making the approximation
    auditable rather than silently substituting it for a refit.
    """

    table = adapt_data(data)
    if group not in table.column_names:
        raise KeyError(f"Group column {group!r} is absent.")
    levels = _group_levels(table[group])
    if len(levels) < 2:
        raise ValueError("Grouping-safe influence analysis requires at least two groups.")

    full_fit = fit_function(table.to_dict()) if baseline is None else baseline
    baseline_parameters = {
        str(name): float(value) for name, value in dict(full_fit.parameters).items()
    }
    if not baseline_parameters:
        raise ValueError("The baseline fit contains no parameters.")
    if getattr(full_fit.convergence, "status", None) == "failed":
        raise RuntimeError("The baseline fit returned failed convergence status.")
    try:
        baseline_objective = float(full_fit.objective)
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("The baseline fit returned a non-numeric objective.") from error
    if not np.isfinite(baseline_objective):
        raise RuntimeError("The baseline fit returned a non-finite objective.")
    parameter_names = tuple(baseline_parameters)
    baseline_rank = _fit_model_rank(full_fit)
    baseline_values = np.asarray(
        [baseline_parameters[name] for name in parameter_names],
        dtype=float,
    )
    if not np.all(np.isfinite(baseline_values)):
        raise RuntimeError("The baseline fit returned non-finite parameter estimates.")
    covariance = getattr(full_fit, "parameter_covariance", None)
    inverse_covariance: np.ndarray | None = None
    cook_distance_status = "unavailable-no-parameter-covariance"
    if covariance is not None:
        matrix = np.asarray(covariance, dtype=float)
        expected = (len(parameter_names), len(parameter_names))
        if matrix.shape != expected:
            raise ValueError(
                f"Baseline parameter covariance has shape {matrix.shape}; expected {expected}."
            )
        if np.any(~np.isfinite(matrix)) or not np.allclose(
            matrix,
            matrix.T,
            rtol=1e-8,
            atol=1e-10,
        ):
            raise ValueError("Baseline parameter covariance must be finite and symmetric.")
        symmetric = (matrix + matrix.T) / 2.0
        eigenvalues = np.linalg.eigvalsh(symmetric)
        scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        if float(eigenvalues[0]) < -1e-10 * scale:
            raise ValueError("Baseline parameter covariance must be positive semidefinite.")
        inverse_covariance = np.linalg.pinv(symmetric)
        cook_distance_status = "available"

    output: dict[str, list[Any]] = {
        "group_index": [],
        "group": [],
        "parameter": [],
        "baseline_estimate": [],
        "full_refit_estimate": [],
        "change": [],
        "absolute_change": [],
        "approximate_estimate": [],
        "approximation_error": [],
        "cook_distance": [],
        "objective_change": [],
        "remaining_rows": [],
        "baseline_model_rank": [],
        "full_refit_model_rank": [],
        "rank_changed": [],
    }
    failures: list[dict[str, Any]] = []
    group_values = np.asarray(table[group])
    for group_index, level in enumerate(levels):
        keep = group_values != level
        sample = {name: values[keep] for name, values in table.items()}
        try:
            refit = fit_function(sample)
            if getattr(refit.convergence, "status", None) == "failed":
                raise RuntimeError("fit returned failed convergence status")
            refit_parameters = {
                str(name): float(value) for name, value in dict(refit.parameters).items()
            }
            refit_rank = _fit_model_rank(refit)
            if set(refit_parameters) != set(parameter_names):
                raise RuntimeError(
                    "parameter set changed after deleting a group "
                    f"(baseline rank {baseline_rank}, refit rank {refit_rank})"
                )
            estimates = np.asarray(
                [refit_parameters[name] for name in parameter_names],
                dtype=float,
            )
            if not np.all(np.isfinite(estimates)):
                raise RuntimeError("refit returned non-finite parameter estimates")
            refit_objective = float(refit.objective)
            if not np.isfinite(refit_objective):
                raise RuntimeError("refit returned a non-finite objective")
        except Exception as error:
            failures.append(
                {
                    "group_index": group_index,
                    "group": repr(level),
                    "phase": "full-refit",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            continue

        approximations = np.full(len(parameter_names), np.nan)
        if approximation is not None:
            try:
                approximation_values = {
                    str(name): float(value)
                    for name, value in approximation(level, full_fit).items()
                }
                if set(approximation_values) != set(parameter_names):
                    raise RuntimeError("approximation parameter set differs from the baseline")
                approximations = np.asarray(
                    [approximation_values[name] for name in parameter_names],
                    dtype=float,
                )
                if not np.all(np.isfinite(approximations)):
                    raise RuntimeError("approximation returned non-finite estimates")
            except Exception as error:
                failures.append(
                    {
                        "group_index": group_index,
                        "group": repr(level),
                        "phase": "approximation",
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
                approximations = np.full(len(parameter_names), np.nan)

        changes = estimates - baseline_values
        cook_distance = (
            max(
                float(changes @ inverse_covariance @ changes / len(parameter_names)),
                0.0,
            )
            if inverse_covariance is not None
            else np.nan
        )
        objective_change = refit_objective - baseline_objective
        for position, name in enumerate(parameter_names):
            output["group_index"].append(group_index)
            output["group"].append(str(level))
            output["parameter"].append(name)
            output["baseline_estimate"].append(baseline_values[position])
            output["full_refit_estimate"].append(estimates[position])
            output["change"].append(changes[position])
            output["absolute_change"].append(abs(changes[position]))
            output["approximate_estimate"].append(approximations[position])
            output["approximation_error"].append(approximations[position] - estimates[position])
            output["cook_distance"].append(cook_distance)
            output["objective_change"].append(objective_change)
            output["remaining_rows"].append(int(np.sum(keep)))
            output["baseline_model_rank"].append(baseline_rank)
            output["full_refit_model_rank"].append(refit_rank)
            output["rank_changed"].append(refit_rank != baseline_rank)

    return GroupInfluenceResult(
        table=DiagnosticTable(
            name="delete_group_influence",
            columns={name: np.asarray(values) for name, values in output.items()},
            metadata={
                "group_column": group,
                "requested_groups": len(levels),
                "parameter_names": list(parameter_names),
                "approximation_compared": approximation is not None,
                "deletion_unit": "complete-group",
                "cook_distance_status": cook_distance_status,
                "model_rank_metric": "fixed_effect_rank",
            },
        ),
        failures=tuple(failures),
        group_column=group,
        requested_groups=len(levels),
    )


def residual_table(
    observed: Sequence[float],
    fitted: Sequence[float],
    *,
    variance: Sequence[float] | float | None = None,
    row_ids: Sequence[Any] | None = None,
    groups: Sequence[Any] | None = None,
) -> DiagnosticTable:
    """Calculate raw and Pearson residuals with source-row reconciliation."""

    y = np.asarray(observed, dtype=float)
    prediction = np.asarray(fitted, dtype=float)
    if y.shape != prediction.shape:
        raise ValueError("Observed and fitted values must have the same shape.")
    raw = y - prediction
    if variance is None:
        denominator = np.full_like(raw, np.nan)
        pearson = np.full_like(raw, np.nan)
    else:
        denominator = np.sqrt(np.broadcast_to(np.asarray(variance, dtype=float), raw.shape))
        pearson = np.divide(
            raw,
            denominator,
            out=np.full_like(raw, np.nan),
            where=denominator > 0,
        )
    columns: dict[str, np.ndarray] = {
        "row_id": np.asarray(row_ids if row_ids is not None else np.arange(len(y))),
        "observed": y,
        "fitted": prediction,
        "raw_residual": raw,
        "pearson_residual": pearson,
    }
    if groups is not None:
        columns["group"] = np.asarray(groups)
    return DiagnosticTable(
        name="residuals",
        columns=columns,
        metadata={
            "definitions": {
                "raw_residual": "observed - fitted",
                "pearson_residual": "(observed - fitted) / sqrt(model variance)",
            }
        },
    )


def _bin_assignments(
    values: np.ndarray, bins: str | int | Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(bins, str):
        if bins != "adaptive":
            raise ValueError("String bins must be 'adaptive'.")
        count = max(1, min(10, int(np.sqrt(len(values)))))
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, count + 1)))
    elif isinstance(bins, int):
        if bins < 1:
            raise ValueError("The number of VPC bins must be positive.")
        edges = np.linspace(float(np.min(values)), float(np.max(values)), bins + 1)
    else:
        edges = np.asarray(bins, dtype=float)
    if edges.size < 2:
        epsilon = max(abs(float(values[0])) * 1e-12, 1e-12)
        edges = np.asarray([values[0] - epsilon, values[0] + epsilon])
    assignments = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
    return assignments, edges


def vpc_table(
    observed: Sequence[float],
    simulated: np.ndarray,
    *,
    independent: Sequence[float] | None = None,
    bins: str | int | Sequence[float] = "adaptive",
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    seed: int | None = None,
    prediction_corrected: bool = False,
) -> DiagnosticTable:
    """Compute an ordinary or prediction-corrected VPC as a tidy table.

    ``simulated`` is shaped ``(replicate, observation)``.  Simulation-interval
    bounds are empirical 2.5% and 97.5% quantiles of within-replicate quantiles.
    """

    obs = np.asarray(observed, dtype=float)
    sims = np.asarray(simulated, dtype=float)
    if sims.ndim != 2 or sims.shape[1] != obs.size:
        raise ValueError("simulated must have shape (replicate, observation).")
    x = np.arange(obs.size, dtype=float) if independent is None else np.asarray(independent)
    if x.shape != obs.shape:
        raise ValueError("The independent variable must align with observed values.")
    assignment, edges = _bin_assignments(x, bins)
    output: dict[str, list[float]] = {
        "bin": [],
        "bin_left": [],
        "bin_right": [],
        "quantile": [],
        "observed": [],
        "simulated_median": [],
        "simulated_lower": [],
        "simulated_upper": [],
        "n_observed": [],
    }
    for bin_number in range(len(edges) - 1):
        selected = assignment == bin_number
        if not np.any(selected):
            continue
        obs_bin = obs[selected]
        sim_bin = sims[:, selected]
        if prediction_corrected:
            typical = np.median(sim_bin, axis=0)
            center = float(np.median(typical))
            scale = np.divide(
                center,
                typical,
                out=np.ones_like(typical),
                where=np.abs(typical) > np.finfo(float).eps,
            )
            obs_bin = obs_bin * scale
            sim_bin = sim_bin * scale[None, :]
        for probability in quantiles:
            replicate_values = np.quantile(sim_bin, probability, axis=1)
            output["bin"].append(float(bin_number))
            output["bin_left"].append(float(edges[bin_number]))
            output["bin_right"].append(float(edges[bin_number + 1]))
            output["quantile"].append(float(probability))
            output["observed"].append(float(np.quantile(obs_bin, probability)))
            output["simulated_median"].append(float(np.median(replicate_values)))
            output["simulated_lower"].append(float(np.quantile(replicate_values, 0.025)))
            output["simulated_upper"].append(float(np.quantile(replicate_values, 0.975)))
            output["n_observed"].append(float(obs_bin.size))
    return DiagnosticTable(
        name="prediction_corrected_vpc" if prediction_corrected else "vpc",
        columns={name: np.asarray(values) for name, values in output.items()},
        metadata={
            "n_replicates": int(sims.shape[0]),
            "quantiles": list(quantiles),
            "bins": edges.tolist(),
            "seed": seed,
            "prediction_corrected": prediction_corrected,
            "interval": [0.025, 0.975],
        },
    )


def covariance_singularity_table(
    blocks: Mapping[str, np.ndarray], *, tolerance: float = 1e-4
) -> DiagnosticTable:
    """Summarize eigenvalue ratios and effective ranks of covariance blocks."""

    names: list[str] = []
    minimum: list[float] = []
    maximum: list[float] = []
    ratios: list[float] = []
    ranks: list[int] = []
    singular: list[bool] = []
    for name, block in blocks.items():
        matrix = np.asarray(block, dtype=float)
        values = np.linalg.eigvalsh((matrix + matrix.T) / 2)
        max_value = float(np.max(values))
        min_value = float(np.min(values))
        ratio = min_value / max_value if max_value > 0 else 0.0
        names.append(name)
        minimum.append(min_value)
        maximum.append(max_value)
        ratios.append(ratio)
        ranks.append(int(np.linalg.matrix_rank(matrix, tol=tolerance * max(max_value, 1))))
        singular.append(bool(ratio < tolerance))
    return DiagnosticTable(
        name="covariance_singularity",
        columns={
            "block": np.asarray(names),
            "min_eigenvalue": np.asarray(minimum),
            "max_eigenvalue": np.asarray(maximum),
            "eigenvalue_ratio": np.asarray(ratios),
            "effective_rank": np.asarray(ranks),
            "singular": np.asarray(singular),
        },
        metadata={"relative_tolerance": tolerance},
    )
