"""Dense repeated-measures Gaussian REML (MMRM) reference backend.

Expected fields in addition to the common backend contract are ``subject`` (or
``subjects``/``group``), ``visit`` (or ``visits``), and a covariance structure
in ``covariance``/``covariance_structure``/``residual``.  Supported structures
are homogeneous, diagonal, unstructured, compound symmetry, AR(1),
heterogeneous AR(1), Toeplitz, heterogeneous Toeplitz, first-order
ante-dependence, and spatial power.  Missing visits are handled by selecting the
observed submatrix of the shared visit covariance.

Satterthwaite denominator degrees of freedom use a finite-difference delta
method for contrast variance and the REML covariance-parameter Hessian.
``kenward-roger`` is intentionally not claimed.  An explicitly requested
``kenward-roger-approximate`` option uses a second-order delta adjustment and is
labelled KR-inspired, not exact KR.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, optimize, special, stats

from ..data import is_missing
from .base import (
    BackendInputError,
    BackendNumericalError,
    BackendUnsupportedError,
    cho_solve,
    convergence_mapping,
    field,
    finite_gradient,
    logdet_from_cholesky,
    make_payload,
    optimizer_covariance,
    prepare_data,
    safe_cholesky,
    select_optimizer_result,
)

_ORDER_DEPENDENT_STRUCTURES = frozenset(
    {
        "ar1",
        "heterogeneous-ar1",
        "toeplitz",
        "heterogeneous-toeplitz",
        "ante-dependence",
        "spatial-power",
        "heterogeneous-spatial-power",
    }
)


def _normalize_structure(value: Any) -> str:
    if value is None:
        return "unstructured"
    if isinstance(value, str):
        name = value
    elif isinstance(value, Mapping):
        name = str(value.get("name", value.get("type", value.get("structure", ""))))
    else:
        name = str(getattr(value, "name", value.__class__.__name__))
    name = name.lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "iid": "homogeneous",
        "identity": "homogeneous",
        "homogeneous": "homogeneous",
        "homogeneous-variance": "homogeneous",
        "diagonal": "diagonal",
        "heterogeneous": "diagonal",
        "un": "unstructured",
        "us": "unstructured",
        "unstructured": "unstructured",
        "cs": "compound-symmetry",
        "compound-symmetry": "compound-symmetry",
        "heterogeneous-cs": "heterogeneous-compound-symmetry",
        "heterogeneous-compound-symmetry": "heterogeneous-compound-symmetry",
        "ar1": "ar1",
        "ar(1)": "ar1",
        "autoregressive": "ar1",
        "heterogeneous-ar1": "heterogeneous-ar1",
        "ar1h": "heterogeneous-ar1",
        "heterogeneous-autoregressive": "heterogeneous-ar1",
        "toeplitz": "toeplitz",
        "toep": "toeplitz",
        "heterogeneous-toeplitz": "heterogeneous-toeplitz",
        "toeph": "heterogeneous-toeplitz",
        "ante-dependence": "ante-dependence",
        "antedependence": "ante-dependence",
        "ad1": "ante-dependence",
        "spatial-power": "spatial-power",
        "spatialpower": "spatial-power",
        "spatial": "spatial-power",
        "heterogeneous-spatial-power": "heterogeneous-spatial-power",
    }
    try:
        return aliases[name]
    except KeyError as exc:
        raise BackendUnsupportedError(
            f"MMRM covariance structure {name!r} is unsupported",
            details={"supported_structures": sorted(set(aliases.values()))},
        ) from exc


def _level_value(raw: Any) -> Any:
    return raw.item() if isinstance(raw, np.generic) else raw


def _level_key(raw: Any) -> tuple[str, str]:
    value = _level_value(raw)
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    return kind, repr(value)


def _stable_factorize(values: Any) -> tuple[NDArray[np.int64], tuple[Any, ...]]:
    """Factorize grouping labels in a permutation-independent order."""

    array = np.asarray(values)
    if array.ndim != 1:
        array = np.ravel(array)
    observed: dict[tuple[str, str], Any] = {}
    row_keys: list[tuple[str, str]] = []
    for raw in array.tolist():
        value = _level_value(raw)
        key = _level_key(value)
        observed.setdefault(key, value)
        row_keys.append(key)
    ordered_keys = tuple(sorted(observed))
    lookup = {key: code for code, key in enumerate(ordered_keys)}
    codes = np.asarray([lookup[key] for key in row_keys], dtype=np.int64)
    return codes, tuple(observed[key] for key in ordered_keys)


def _declared_categories(values: Any) -> tuple[tuple[Any, ...], bool | None]:
    """Read pandas-like categorical metadata without importing pandas."""

    try:
        categorical = values.cat
    except (AttributeError, TypeError):
        categorical = values
    categories = getattr(categorical, "categories", None)
    if categories is None:
        return (), None
    try:
        levels = tuple(_level_value(item) for item in categories.tolist())
    except AttributeError:
        levels = tuple(_level_value(item) for item in categories)
    return levels, bool(getattr(categorical, "ordered", False))


def _visit_field_name(source: Any, covariance: Any) -> str:
    if isinstance(covariance, Mapping):
        index = covariance.get("index")
    else:
        index = getattr(covariance, "index", None)
    if isinstance(index, str) and index:
        return index
    if isinstance(source, Mapping):
        for candidate in ("visit", "visits", "visit_id", "visit_ids"):
            if candidate in source:
                return candidate
    return "visit"


def _audit_categories(source: Any, visit_name: str) -> tuple[tuple[Any, ...], bool | None]:
    audit = field(source, "audit", default=None)
    levels_by_factor = getattr(audit, "factor_levels", {})
    ordering_by_factor = getattr(audit, "factor_ordered", {})
    if not isinstance(levels_by_factor, Mapping) or visit_name not in levels_by_factor:
        return (), None
    levels = tuple(_level_value(item) for item in levels_by_factor[visit_name])
    ordered = (
        bool(ordering_by_factor[visit_name])
        if isinstance(ordering_by_factor, Mapping) and visit_name in ordering_by_factor
        else None
    )
    return levels, ordered


def _numeric_levels(levels: Sequence[Any]) -> NDArray[np.float64] | None:
    if any(isinstance(value, (bool, np.bool_)) for value in levels):
        return None
    try:
        numeric = np.asarray(levels, dtype=float)
    except (TypeError, ValueError):
        return None
    if numeric.ndim != 1 or np.any(~np.isfinite(numeric)):
        return None
    return numeric


def _validate_declared_levels(
    declared: Sequence[Any],
    observed: Mapping[tuple[str, str], Any],
    *,
    label: str,
) -> tuple[Any, ...]:
    declared_by_key: dict[tuple[str, str], Any] = {}
    for raw in declared:
        value = _level_value(raw)
        key = _level_key(value)
        if key in declared_by_key:
            raise BackendInputError(f"{label} contains duplicate visit level {value!r}")
        declared_by_key[key] = value
    missing = [value for key, value in observed.items() if key not in declared_by_key]
    if missing:
        raise BackendInputError(
            f"{label} does not include every observed visit",
            details={"missing_levels": [repr(value) for value in missing]},
        )
    # Globally unobserved declared categories have no estimable covariance
    # parameters. Preserve their relative order but restrict the fitted axis to
    # levels represented in the analysis data.
    return tuple(declared_by_key[key] for key in declared_by_key if key in observed)


@dataclass(slots=True)
class _VisitAxis:
    codes: NDArray[np.int64]
    levels: tuple[Any, ...]
    times: NDArray[np.float64]
    order_source: str
    declared_levels: tuple[Any, ...]


def _prepare_visit_axis(
    source: Any,
    visit_raw: Any,
    visit_times_raw: Any,
    *,
    covariance: Any,
    structure: str,
    n_observations: int,
) -> _VisitAxis:
    """Resolve a row-order-independent, scientifically explicit visit axis."""

    raw_array = np.asarray(visit_raw)
    if raw_array.ndim != 1:
        raw_array = np.ravel(raw_array)
    if raw_array.size != n_observations:
        raise BackendInputError("visit vector must match response length")

    observed: dict[tuple[str, str], Any] = {}
    row_keys: list[tuple[str, str]] = []
    for position, raw in enumerate(raw_array.tolist()):
        value = _level_value(raw)
        if is_missing(value):
            raise BackendInputError(
                "MMRM visit labels cannot be missing",
                details={"row": position},
            )
        key = _level_key(value)
        observed.setdefault(key, value)
        row_keys.append(key)
    if not observed:
        raise BackendInputError("MMRM requires at least one visit")

    explicit_raw = field(
        source,
        "visit_order",
        "ordered_visit_levels",
        "visit_levels",
        default=None,
    )
    explicit_levels = (
        ()
        if explicit_raw is None
        else tuple(_level_value(item) for item in np.asarray(explicit_raw).reshape(-1).tolist())
    )
    categorical_levels, categorical_ordered = _declared_categories(visit_raw)
    visit_name = _visit_field_name(source, covariance)
    if not categorical_levels:
        categorical_levels, categorical_ordered = _audit_categories(source, visit_name)

    declared_levels: tuple[Any, ...] = ()
    semantically_ordered = False
    if explicit_levels:
        base_levels = _validate_declared_levels(explicit_levels, observed, label="visit_order")
        declared_levels = explicit_levels
        base_source = "explicit-visit-order"
        semantically_ordered = True
    elif categorical_levels:
        base_levels = _validate_declared_levels(
            categorical_levels,
            observed,
            label=f"declared levels for {visit_name!r}",
        )
        declared_levels = categorical_levels
        semantically_ordered = categorical_ordered is True
        base_source = (
            "ordered-categorical-levels" if semantically_ordered else "unordered-categorical-levels"
        )
    else:
        observed_levels = tuple(observed.values())
        numeric_observed = _numeric_levels(observed_levels)
        if numeric_observed is not None:
            order = np.argsort(numeric_observed, kind="stable")
            sorted_numeric = numeric_observed[order]
            if sorted_numeric.size > 1 and np.any(np.diff(sorted_numeric) <= 0):
                raise BackendInputError(
                    "numeric visit labels must identify distinct ordered visits"
                )
            base_levels = tuple(observed_levels[index] for index in order.tolist())
            base_source = "ascending-numeric-visit-labels"
            semantically_ordered = True
        else:
            base_levels = tuple(observed[key] for key in sorted(observed))
            base_source = "deterministic-lexical-visit-labels"

    time_by_key: dict[tuple[str, str], float] | None = None
    if visit_times_raw is not None and isinstance(visit_times_raw, Mapping):
        time_by_key = {}
        for raw_level, raw_time in visit_times_raw.items():
            key = _level_key(raw_level)
            if key not in observed:
                continue
            try:
                value = float(raw_time)
            except (TypeError, ValueError) as exc:
                raise BackendInputError("visit_times mapping values must be numeric") from exc
            if not np.isfinite(value):
                raise BackendInputError("visit times must be finite")
            time_by_key[key] = value
        missing_time = [value for key, value in observed.items() if key not in time_by_key]
        if missing_time:
            raise BackendInputError(
                "visit_times mapping does not include every observed visit",
                details={"missing_levels": [repr(value) for value in missing_time]},
            )
    elif visit_times_raw is not None:
        try:
            time_values = np.asarray(visit_times_raw, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            time_values = np.empty(0)
        if time_values.size:
            if np.any(~np.isfinite(time_values)):
                raise BackendInputError("visit times must be finite")
            if time_values.size == n_observations:
                time_by_key = {}
                for key, value in zip(row_keys, time_values.tolist(), strict=True):
                    previous = time_by_key.setdefault(key, float(value))
                    if previous != float(value):
                        raise BackendInputError(
                            "each visit must map to exactly one numeric visit time",
                            details={"visit": repr(observed[key])},
                        )
            elif time_values.size == len(base_levels):
                if not semantically_ordered:
                    raise BackendInputError(
                        "visit_times with one value per visit requires visit_order, "
                        "ordered categorical levels, or numeric visit labels"
                    )
                time_by_key = {
                    _level_key(level): float(value)
                    for level, value in zip(base_levels, time_values.tolist(), strict=True)
                }
            else:
                raise BackendInputError("visit_times must have one value per visit or observation")

    if time_by_key is not None:
        ordered_pairs = sorted(
            ((time_by_key[_level_key(level)], level) for level in base_levels),
            key=lambda item: item[0],
        )
        ordered_times = np.asarray([item[0] for item in ordered_pairs], dtype=float)
        if ordered_times.size > 1 and np.any(np.diff(ordered_times) <= 0):
            raise BackendInputError(
                "distinct visits must map to strictly increasing numeric visit times"
            )
        levels = tuple(item[1] for item in ordered_pairs)
        visit_times = ordered_times
        order_source = "ascending-explicit-visit-times"
    else:
        levels = base_levels
        numeric_levels = _numeric_levels(levels)
        if numeric_levels is not None:
            visit_times = numeric_levels
        else:
            visit_times = np.arange(len(levels), dtype=float)
        order_source = base_source
        if structure in _ORDER_DEPENDENT_STRUCTURES and not semantically_ordered:
            raise BackendInputError(
                f"MMRM covariance structure {structure!r} requires an explicit visit order "
                "for nonnumeric labels",
                details={
                    "provide": [
                        "visit_order",
                        "ordered categorical visit levels",
                        "numeric visit_times",
                    ],
                    "observed_levels": [repr(level) for level in levels],
                },
            )

    level_codes = {_level_key(level): code for code, level in enumerate(levels)}
    codes = np.asarray([level_codes[key] for key in row_keys], dtype=np.int64)
    return _VisitAxis(
        codes=codes,
        levels=levels,
        times=visit_times,
        order_source=order_source,
        declared_levels=declared_levels,
    )


def _pacf_to_autocorrelation(raw: ArrayLike) -> NDArray[np.float64]:
    """Map unconstrained partial autocorrelations to a positive Toeplitz row."""

    pacf = np.tanh(np.asarray(raw, dtype=float))
    order = pacf.size
    autocorrelation = np.ones(order + 1)
    previous = np.zeros(order + 1)
    for current_order in range(1, order + 1):
        current = np.zeros(order + 1)
        current[current_order] = pacf[current_order - 1]
        for lag in range(1, current_order):
            current[lag] = previous[lag] - pacf[current_order - 1] * previous[current_order - lag]
        autocorrelation[current_order] = sum(
            current[lag] * autocorrelation[current_order - lag]
            for lag in range(1, current_order + 1)
        )
        previous = current
    return autocorrelation


@dataclass(slots=True)
class _VisitCovariance:
    structure: str
    visits: tuple[Any, ...]
    visit_times: NDArray[np.float64]

    @property
    def q(self) -> int:
        return len(self.visits)

    @property
    def size(self) -> int:
        q = self.q
        if self.structure == "homogeneous":
            return 1
        if self.structure == "diagonal":
            return q
        if self.structure == "unstructured":
            return q * (q + 1) // 2
        if self.structure in {"compound-symmetry", "ar1", "spatial-power"}:
            return 1 if q == 1 else 2
        if self.structure in {
            "heterogeneous-compound-symmetry",
            "heterogeneous-ar1",
            "heterogeneous-spatial-power",
        }:
            return q if q == 1 else q + 1
        if self.structure == "toeplitz":
            return q
        if self.structure in {"heterogeneous-toeplitz", "ante-dependence"}:
            return q + max(q - 1, 0)
        raise AssertionError(self.structure)

    def initial(self, scale: float) -> NDArray[np.float64]:
        q = self.q
        log_scale = np.log(max(scale, 1e-4))
        if self.structure == "homogeneous":
            return np.array([log_scale])
        if self.structure == "diagonal":
            return np.full(q, log_scale)
        if self.structure == "unstructured":
            output: list[float] = []
            for row in range(q):
                for col in range(row + 1):
                    output.append(log_scale if row == col else 0.0)
            return np.asarray(output)
        if self.structure in {"compound-symmetry", "ar1"}:
            return np.array([log_scale]) if q == 1 else np.array([log_scale, 0.0])
        if self.structure == "spatial-power":
            return np.array([log_scale]) if q == 1 else np.array([log_scale, 0.0])
        if self.structure in {
            "heterogeneous-compound-symmetry",
            "heterogeneous-ar1",
            "heterogeneous-spatial-power",
        }:
            return np.concatenate([np.full(q, log_scale), np.zeros(0 if q == 1 else 1)])
        if self.structure == "toeplitz":
            return np.concatenate([[log_scale], np.zeros(q - 1)])
        if self.structure in {"heterogeneous-toeplitz", "ante-dependence"}:
            return np.concatenate([np.full(q, log_scale), np.zeros(q - 1)])
        raise AssertionError(self.structure)

    def _std_and_correlation(
        self, theta: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        theta = np.asarray(theta, dtype=float)
        q = self.q
        if theta.size != self.size:
            raise ValueError("wrong MMRM covariance parameter length")
        if self.structure == "homogeneous":
            return np.full(q, np.exp(theta[0])), np.eye(q)
        if self.structure == "diagonal":
            return np.exp(theta), np.eye(q)
        if self.structure == "unstructured":
            chol = np.zeros((q, q))
            cursor = 0
            for row in range(q):
                for col in range(row + 1):
                    chol[row, col] = np.exp(theta[cursor]) if row == col else theta[cursor]
                    cursor += 1
            covariance = chol @ chol.T
            std = np.sqrt(np.diag(covariance))
            return std, covariance / std[:, None] / std[None, :]
        if self.structure in {"compound-symmetry", "heterogeneous-compound-symmetry"}:
            homogeneous = self.structure == "compound-symmetry"
            std = np.full(q, np.exp(theta[0])) if homogeneous else np.exp(theta[:q])
            if q == 1:
                return std, np.eye(1)
            raw = theta[1] if homogeneous else theta[q]
            lower = -1.0 / (q - 1) + 1e-7
            rho = lower + (1.0 - lower) * special.expit(raw)
            correlation = np.full((q, q), rho)
            np.fill_diagonal(correlation, 1.0)
            return std, correlation
        if self.structure in {"ar1", "heterogeneous-ar1"}:
            homogeneous = self.structure == "ar1"
            std = np.full(q, np.exp(theta[0])) if homogeneous else np.exp(theta[:q])
            rho = 0.0 if q == 1 else np.tanh(theta[1] if homogeneous else theta[q])
            lag = np.abs(np.subtract.outer(np.arange(q), np.arange(q)))
            return std, rho**lag
        if self.structure in {"toeplitz", "heterogeneous-toeplitz"}:
            homogeneous = self.structure == "toeplitz"
            std = np.full(q, np.exp(theta[0])) if homogeneous else np.exp(theta[:q])
            raw = theta[1:] if homogeneous else theta[q:]
            first_row = _pacf_to_autocorrelation(raw)
            lag = np.abs(np.subtract.outer(np.arange(q), np.arange(q)))
            return std, first_row[lag]
        if self.structure == "ante-dependence":
            std = np.exp(theta[:q])
            adjacent = np.tanh(theta[q:])
            correlation = np.eye(q)
            for row in range(q):
                for col in range(row):
                    correlation[row, col] = correlation[col, row] = np.prod(adjacent[col:row])
            return std, correlation
        if self.structure in {"spatial-power", "heterogeneous-spatial-power"}:
            homogeneous = self.structure == "spatial-power"
            std = np.full(q, np.exp(theta[0])) if homogeneous else np.exp(theta[:q])
            if q == 1:
                return std, np.eye(1)
            rate = np.exp(theta[1] if homogeneous else theta[q])
            distance = np.abs(np.subtract.outer(self.visit_times, self.visit_times))
            return std, np.exp(-rate * distance)
        raise AssertionError(self.structure)

    def matrix(self, theta: ArrayLike) -> NDArray[np.float64]:
        std, correlation = self._std_and_correlation(theta)
        covariance = std[:, None] * correlation * std[None, :]
        safe_cholesky(covariance)
        return covariance

    def natural(self, theta: ArrayLike) -> dict[str, float]:
        std, correlation = self._std_and_correlation(theta)
        homogeneous = self.structure in {
            "homogeneous",
            "compound-symmetry",
            "ar1",
            "toeplitz",
            "spatial-power",
        }
        output = (
            {"residual_sd": float(std[0])}
            if homogeneous
            else {
                f"sd(visit={visit!s})": float(std[index]) for index, visit in enumerate(self.visits)
            }
        )
        if self.structure == "homogeneous":
            return output
        if (
            self.structure in {"compound-symmetry", "heterogeneous-compound-symmetry"}
            and self.q > 1
        ):
            output["correlation"] = float(correlation[0, 1])
        elif self.structure in {"ar1", "heterogeneous-ar1"} and self.q > 1:
            output["ar1_correlation"] = float(correlation[0, 1])
        elif self.structure in {"spatial-power", "heterogeneous-spatial-power"} and self.q > 1:
            # Report correlation at one time unit, not an opaque log-rate.
            distance_one = float(
                np.exp(-np.exp(theta[1] if self.structure == "spatial-power" else theta[self.q]))
            )
            output["spatial_correlation_at_unit_distance"] = distance_one
        elif self.structure in {"toeplitz", "heterogeneous-toeplitz"}:
            for lag in range(1, self.q):
                output[f"correlation_lag_{lag}"] = float(correlation[0, lag])
        elif self.structure in {"unstructured", "ante-dependence"}:
            for row in range(1, self.q):
                for col in range(row):
                    output[f"corr(visit={self.visits[row]!s},visit={self.visits[col]!s})"] = float(
                        correlation[row, col]
                    )
        return output

    def unconstrained_names(self) -> list[str]:
        q = self.q
        if self.structure == "homogeneous":
            return ["log_residual_sd"]
        if self.structure == "diagonal":
            return [f"log_sd(visit={visit!s})" for visit in self.visits]
        if self.structure == "unstructured":
            names: list[str] = []
            for row in range(q):
                for col in range(row + 1):
                    if row == col:
                        names.append(f"log_chol(visit={self.visits[row]!s})")
                    else:
                        names.append(f"chol(visit={self.visits[row]!s},visit={self.visits[col]!s})")
            return names
        heterogeneous = (
            self.structure.startswith("heterogeneous") or self.structure == "ante-dependence"
        )
        names = (
            [f"log_sd(visit={visit!s})" for visit in self.visits]
            if heterogeneous
            else ["log_residual_sd"]
        )
        remaining = self.size - len(names)
        if self.structure in {"toeplitz", "heterogeneous-toeplitz"}:
            names.extend(
                f"raw_partial_autocorrelation_lag_{lag}" for lag in range(1, remaining + 1)
            )
        elif self.structure == "ante-dependence":
            names.extend(f"raw_adjacent_correlation_{lag}_{lag + 1}" for lag in range(remaining))
        elif self.structure in {"spatial-power", "heterogeneous-spatial-power"} and remaining:
            names.append("log_spatial_decay")
        elif remaining:
            names.append("raw_correlation")
        return names


@dataclass(slots=True)
class _MMRMEvaluation:
    log_likelihood: float
    beta: NDArray[np.float64]
    beta_covariance: NDArray[np.float64]
    residual: NDArray[np.float64]
    covariance: NDArray[np.float64]
    cholesky: NDArray[np.float64]
    rank: int
    jitter: float


def linear_inference(
    beta: ArrayLike,
    covariance: ArrayLike,
    matrix: ArrayLike,
    *,
    names: Sequence[str] | None = None,
    degrees_of_freedom: ArrayLike | float = np.inf,
    confidence_level: float = 0.95,
    method: str = "Wald",
) -> dict[str, list[Any]]:
    """Build a tidy table for linear EMM/contrast estimates."""

    beta = np.asarray(beta, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[1] != beta.size:
        raise BackendInputError("inference matrix columns must match fixed effects")
    estimates = matrix @ beta
    variances = np.einsum("ij,jk,ik->i", matrix, covariance, matrix)
    standard_errors = np.sqrt(np.maximum(variances, 0.0))
    df = np.broadcast_to(np.asarray(degrees_of_freedom, dtype=float), estimates.shape)
    alpha = 1 - confidence_level
    critical = np.where(
        np.isfinite(df), stats.t.ppf(1 - alpha / 2, df), stats.norm.ppf(1 - alpha / 2)
    )
    labels = (
        list(names) if names is not None else [f"L{index + 1}" for index in range(len(estimates))]
    )
    if len(labels) != len(estimates):
        raise BackendInputError("inference names length does not match matrix rows")
    return {
        "name": labels,
        "estimate": estimates.tolist(),
        "standard_error": standard_errors.tolist(),
        "degrees_of_freedom": df.tolist(),
        "lower_confidence_limit": (estimates - critical * standard_errors).tolist(),
        "upper_confidence_limit": (estimates + critical * standard_errors).tolist(),
        "method": [method] * len(estimates),
    }


class MMRMBackend:
    """Dense subject-block MMRM REML engine."""

    name = "mmrm"

    def fit(
        self,
        data: Any,
        *,
        covariance: Any = None,
        method: str = "REML",
        df_method: str = "satterthwaite",
        confidence_level: float = 0.95,
        maxiter: int = 1_000,
        tolerance: float = 1e-8,
        compute_hessian: bool = True,
        **options: Any,
    ) -> dict[str, Any]:
        compiled = prepare_data(data)
        if compiled.random_blocks:
            raise BackendUnsupportedError(
                "dedicated MMRM currently models within-subject covariance without random-effect blocks",
                suggested_engines=("lmm",),
            )
        method = method.upper()
        if method not in {"REML", "ML"}:
            raise BackendInputError("MMRM method must be REML or ML")
        source = compiled.source
        subject_raw = field(
            source, "subject", "subjects", "subject_id", "subject_ids", "group", "groups"
        )
        visit_raw = field(source, "visit", "visits", "visit_id", "visit_ids")
        subject_codes, subject_levels = _stable_factorize(subject_raw)
        if subject_codes.size != compiled.n_obs:
            raise BackendInputError("subject vector must match response length")

        if covariance is None:
            covariance = field(
                source,
                "covariance_structure",
                "covariance",
                default=compiled.residual_covariance,
            )
        explicit_covariance_input = isinstance(covariance, (np.ndarray, list, tuple))
        ordering_structure = (
            "explicit-fixed" if explicit_covariance_input else _normalize_structure(covariance)
        )
        visit_axis = _prepare_visit_axis(
            source,
            visit_raw,
            field(source, "visit_times", "times", "time", default=None),
            covariance=covariance,
            structure=ordering_structure,
            n_observations=compiled.n_obs,
        )
        visit_codes = visit_axis.codes
        visit_levels = visit_axis.levels
        visit_times = visit_axis.times
        q = len(visit_levels)
        canonical_order = np.lexsort((visit_codes, subject_codes))
        observed_pairs = list(zip(subject_codes.tolist(), visit_codes.tolist(), strict=True))
        if len(set(observed_pairs)) != len(observed_pairs):
            raise BackendInputError(
                "MMRM reference engine requires at most one observation per subject and visit"
            )
        explicit_covariance: NDArray[np.float64] | None = None
        if explicit_covariance_input:
            candidate = np.asarray(covariance, dtype=float)
            if candidate.shape == (compiled.n_obs, compiled.n_obs):
                explicit_covariance = (candidate + candidate.T) / 2
                try:
                    safe_cholesky(explicit_covariance)
                except linalg.LinAlgError as exc:
                    raise BackendInputError(
                        "explicit MMRM covariance must be positive definite"
                    ) from exc
                structure_name = "explicit-fixed"
                visit_covariance = None
                theta0 = np.zeros(0)
            else:
                raise BackendInputError(
                    "explicit MMRM covariance must be an n_observation square matrix"
                )
        else:
            structure_name = _normalize_structure(covariance)
            visit_covariance = _VisitCovariance(
                structure_name, visit_levels, np.asarray(visit_times, dtype=float)
            )
            response_scale = max(
                (float(np.std(compiled.y[canonical_order], ddof=1)) if compiled.n_obs > 1 else 1.0),
                1e-3,
            )
            theta0 = visit_covariance.initial(response_scale)

        subject_rows = [
            np.flatnonzero(subject_codes == code) for code in range(len(subject_levels))
        ]

        def full_covariance(
            theta: ArrayLike,
        ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
            if explicit_covariance is not None:
                covariance_matrix = explicit_covariance.copy()
                visit_matrix = None
            else:
                assert visit_covariance is not None
                visit_matrix = visit_covariance.matrix(theta)
                covariance_matrix = np.zeros((compiled.n_obs, compiled.n_obs))
                for rows in subject_rows:
                    codes = visit_codes[rows]
                    covariance_matrix[np.ix_(rows, rows)] = visit_matrix[np.ix_(codes, codes)]
            scaling = 1 / np.sqrt(compiled.weights)
            covariance_matrix = scaling[:, None] * covariance_matrix * scaling[None, :]
            return covariance_matrix, visit_matrix

        def evaluate(theta: ArrayLike) -> _MMRMEvaluation:
            covariance_matrix, _ = full_covariance(theta)
            numerical_covariance = covariance_matrix[np.ix_(canonical_order, canonical_order)]
            numerical_y = compiled.y[canonical_order]
            numerical_X = compiled.X[canonical_order]
            cholesky, jitter = safe_cholesky(numerical_covariance)
            yw = linalg.solve_triangular(cholesky, numerical_y, lower=True, check_finite=False)
            Xw = linalg.solve_triangular(cholesky, numerical_X, lower=True, check_finite=False)
            beta, _, rank, singular_values = np.linalg.lstsq(Xw, yw, rcond=None)
            residual = compiled.y - compiled.X @ beta
            numerical_residual = residual[canonical_order]
            quadratic = float(numerical_residual @ cho_solve(cholesky, numerical_residual))
            logdet_covariance = logdet_from_cholesky(cholesky)
            information = Xw.T @ Xw
            beta_covariance = np.linalg.pinv(information, rcond=1e-10)
            logdet_information = float(np.log(singular_values[:rank] ** 2).sum()) if rank else 0.0
            if method == "REML":
                log_likelihood = -0.5 * (
                    (compiled.n_obs - rank) * np.log(2 * np.pi)
                    + logdet_covariance
                    + logdet_information
                    + quadratic
                )
            else:
                log_likelihood = -0.5 * (
                    compiled.n_obs * np.log(2 * np.pi) + logdet_covariance + quadratic
                )
            return _MMRMEvaluation(
                log_likelihood=float(log_likelihood),
                beta=beta,
                beta_covariance=beta_covariance,
                residual=residual,
                covariance=covariance_matrix,
                cholesky=cholesky,
                rank=int(rank),
                jitter=jitter,
            )

        def objective(theta: ArrayLike) -> float:
            try:
                value = -evaluate(theta).log_likelihood
                return value if np.isfinite(value) else 1e100
            except (ValueError, FloatingPointError, linalg.LinAlgError):
                return 1e100

        if theta0.size:
            bounds = [(-12.0, 12.0)] * theta0.size
            result = optimize.minimize(
                objective,
                theta0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": int(maxiter), "ftol": tolerance, "gtol": tolerance},
            )
            initial_gradient = np.asarray(getattr(result, "jac", np.array([np.inf])))
            if (
                not result.success
                or not np.isfinite(result.fun)
                or np.max(np.abs(initial_gradient)) > max(np.sqrt(tolerance), 1e-5)
            ):
                rescue = optimize.minimize(
                    objective,
                    np.clip(result.x if np.all(np.isfinite(result.x)) else theta0, -12, 12),
                    method="Powell",
                    bounds=bounds,
                    options={"maxiter": int(maxiter), "xtol": tolerance, "ftol": tolerance},
                )
                refined = optimize.minimize(
                    objective,
                    rescue.x,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": int(maxiter), "ftol": tolerance, "gtol": tolerance},
                )
                result = select_optimizer_result(
                    refined,
                    rescue,
                    objective_tolerance=1e-8,
                )
            theta_hat = np.asarray(result.x)
            success = bool(result.success and np.isfinite(result.fun))
            optimizer_message = str(result.message)
            iterations = int(getattr(result, "nit", 0))
            evaluations = int(getattr(result, "nfev", 0))
            gradient = getattr(result, "jac", None)
            if gradient is None or np.any(~np.isfinite(gradient)):
                gradient = finite_gradient(objective, theta_hat)
        else:
            theta_hat = theta0
            success = True
            optimizer_message = "Explicit covariance required no optimization"
            iterations = evaluations = 0
            gradient = np.zeros(0)

        try:
            final = evaluate(theta_hat)
        except (ValueError, linalg.LinAlgError) as exc:
            raise BackendNumericalError("final MMRM covariance is not positive definite") from exc

        theta_covariance, hessian_positive, curvature_source = optimizer_covariance(
            objective,
            theta_hat,
            result if theta_hat.size else None,
            compute_hessian=compute_hessian,
            finite_difference_limit=25,
        )

        natural_parameters = {
            name: float(value) for name, value in zip(compiled.fixed_names, final.beta, strict=True)
        }
        unconstrained_parameters = dict(natural_parameters)
        covariance_names: list[str] = []
        raw_names: list[str] = []
        if visit_covariance is not None:
            covariance_values = visit_covariance.natural(theta_hat)
            natural_parameters.update(covariance_values)
            covariance_names = list(covariance_values)
            raw_names = visit_covariance.unconstrained_names()
            unconstrained_parameters.update(
                {name: float(value) for name, value in zip(raw_names, theta_hat, strict=True)}
            )

            def natural_covariance_vector(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                return np.asarray(list(visit_covariance.natural(theta).values()))

            jacobian = np.empty((len(covariance_names), theta_hat.size))
            for column in range(theta_hat.size):
                step = 1e-5 * max(1.0, abs(theta_hat[column]))
                delta = np.zeros(theta_hat.size)
                delta[column] = step
                jacobian[:, column] = (
                    natural_covariance_vector(theta_hat + delta)
                    - natural_covariance_vector(theta_hat - delta)
                ) / (2 * step)
            natural_covariance_covariance = jacobian @ theta_covariance @ jacobian.T
        else:
            natural_covariance_covariance = np.zeros((0, 0))
        parameter_covariance = linalg.block_diag(
            final.beta_covariance, natural_covariance_covariance
        )

        residual_df = max(compiled.n_obs - final.rank, 1)

        def contrast_df(
            matrix: NDArray[np.float64], beta_covariance: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            if df_method_normalized == "residual" or not theta_hat.size:
                return np.full(matrix.shape[0], float(residual_df))
            output = np.empty(matrix.shape[0])
            for row_index, contrast in enumerate(matrix):
                variance = float(contrast @ beta_covariance @ contrast)
                derivative = np.empty(theta_hat.size)
                for column in range(theta_hat.size):
                    step = 2e-4 * max(1.0, abs(theta_hat[column]))
                    delta = np.zeros(theta_hat.size)
                    delta[column] = step
                    plus = evaluate(theta_hat + delta).beta_covariance
                    minus = evaluate(theta_hat - delta).beta_covariance
                    derivative[column] = float(contrast @ (plus - minus) @ contrast / (2 * step))
                denominator = float(derivative @ theta_covariance @ derivative)
                output[row_index] = (
                    2 * variance**2 / denominator
                    if denominator > np.finfo(float).eps and variance > 0
                    else residual_df
                )
            return np.maximum(output, 1.0)

        df_method_normalized = df_method.lower().replace("_", "-")
        if df_method_normalized in {"satterthwaite", "satt"}:
            df_method_normalized = "satterthwaite"
            inference_covariance = final.beta_covariance
            inference_label = "Satterthwaite delta-method"
        elif df_method_normalized in {"residual", "residual-df"}:
            df_method_normalized = "residual"
            inference_covariance = final.beta_covariance
            inference_label = "residual degrees of freedom"
        elif df_method_normalized in {"kenward-roger", "kr", "kenward-roger-exact"}:
            raise BackendUnsupportedError(
                "exact Kenward-Roger adjustment is not implemented by the reference MMRM engine",
                details={
                    "available": [
                        "satterthwaite",
                        "residual",
                        "kenward-roger-approximate",
                    ]
                },
            )
        elif df_method_normalized in {
            "kenward-roger-approximate",
            "kr-approximate",
            "kr-inspired",
        }:
            df_method_normalized = "kenward-roger-approximate"
            if theta_hat.size > 12:
                raise BackendUnsupportedError(
                    "KR-inspired second-order delta adjustment is limited to 12 covariance parameters"
                )
            # E[B(theta)] second-order delta approximation.  This captures
            # covariance-parameter uncertainty but is not the KR information
            # adjustment and is labelled accordingly.
            correction = np.zeros_like(final.beta_covariance)
            step = 2e-3 * np.maximum(1.0, np.abs(theta_hat))
            for first in range(theta_hat.size):
                e_first = np.zeros(theta_hat.size)
                e_first[first] = step[first]
                diagonal_second = (
                    evaluate(theta_hat + e_first).beta_covariance
                    - 2 * final.beta_covariance
                    + evaluate(theta_hat - e_first).beta_covariance
                ) / step[first] ** 2
                correction += 0.5 * theta_covariance[first, first] * diagonal_second
                for second in range(first):
                    e_second = np.zeros(theta_hat.size)
                    e_second[second] = step[second]
                    cross_second = (
                        evaluate(theta_hat + e_first + e_second).beta_covariance
                        - evaluate(theta_hat + e_first - e_second).beta_covariance
                        - evaluate(theta_hat - e_first + e_second).beta_covariance
                        + evaluate(theta_hat - e_first - e_second).beta_covariance
                    ) / (4 * step[first] * step[second])
                    correction += theta_covariance[first, second] * cross_second
            inference_covariance = final.beta_covariance + correction
            inference_covariance = (inference_covariance + inference_covariance.T) / 2
            eigenvalues, eigenvectors = np.linalg.eigh(inference_covariance)
            inference_covariance = (
                eigenvectors * np.maximum(eigenvalues, np.finfo(float).eps)
            ) @ eigenvectors.T
            inference_label = (
                "second-order delta covariance adjustment (KR-inspired; not exact Kenward-Roger)"
            )
        else:
            raise BackendInputError(
                "df_method must be satterthwaite, residual, kenward-roger, "
                "or kenward-roger-approximate"
            )

        def inference_spec(*names: str) -> tuple[NDArray[np.float64] | None, list[str] | None]:
            raw = field(source, *names, default=None)
            if raw is None:
                return None, None
            if isinstance(raw, Mapping):
                matrix = np.asarray(raw.get("matrix", raw.get("L")), dtype=float)
                labels = raw.get("names", raw.get("labels"))
                return matrix, None if labels is None else [str(item) for item in labels]
            return np.asarray(raw, dtype=float), None

        emm_matrix, emm_names = inference_spec(
            "emm_matrix", "emmean_matrix", "estimated_marginal_mean_matrix"
        )
        contrast_raw = field(source, "contrasts", "contrast_matrix", default=None)
        contrast_matrix: NDArray[np.float64] | None
        contrast_names: list[str] | None
        if contrast_raw is None:
            contrast_matrix = None
            contrast_names = None
        elif isinstance(contrast_raw, Mapping) and "matrix" not in contrast_raw:
            contrast_names = [str(key) for key in contrast_raw]
            contrast_matrix = np.vstack(
                [np.asarray(value, dtype=float) for value in contrast_raw.values()]
            )
        elif isinstance(contrast_raw, Mapping):
            contrast_matrix = np.asarray(contrast_raw.get("matrix"), dtype=float)
            labels = contrast_raw.get("names", contrast_raw.get("labels"))
            contrast_names = None if labels is None else [str(item) for item in labels]
        else:
            contrast_matrix = np.asarray(contrast_raw, dtype=float)
            contrast_names = None

        def make_inference(
            matrix: NDArray[np.float64] | None, names: list[str] | None
        ) -> dict[str, list[Any]] | None:
            if matrix is None:
                return None
            if matrix.ndim == 1:
                matrix = matrix[None, :]
            degrees = contrast_df(matrix, inference_covariance)
            return linear_inference(
                final.beta,
                inference_covariance,
                matrix,
                names=names,
                degrees_of_freedom=degrees,
                confidence_level=confidence_level,
                method=inference_label,
            )

        emm_table = make_inference(emm_matrix, emm_names)
        contrast_table = make_inference(contrast_matrix, contrast_names)
        _, visit_matrix_hat = full_covariance(theta_hat)
        covariance_eigenvalues = np.linalg.eigvalsh(
            visit_matrix_hat if visit_matrix_hat is not None else final.covariance
        )
        covariance_ratio = float(covariance_eigenvalues[0] / covariance_eigenvalues[-1])
        singular = covariance_ratio < 1e-7
        warnings: list[dict[str, Any]] = []
        if singular:
            warnings.append(
                {
                    "code": "COV-SINGULAR-001",
                    "severity": "warning",
                    "message": "MMRM covariance is singular or near-singular",
                    "details": {"eigenvalue_ratio": covariance_ratio},
                }
            )
        if df_method_normalized == "kenward-roger-approximate":
            warnings.append(
                {
                    "code": "MMRM-KR-APPROX-001",
                    "severity": "warning",
                    "message": "requested adjustment is KR-inspired and is not exact Kenward-Roger",
                    "details": {"method": inference_label},
                }
            )
        convergence = convergence_mapping(
            success=success,
            message=optimizer_message,
            iterations=iterations,
            function_evaluations=evaluations,
            gradient=gradient,
            hessian_positive_definite=hessian_positive,
            singular=singular,
            warnings=warnings,
            extra={
                "fixed_effect_rank": final.rank,
                "subjects": len(subject_levels),
                "visits": q,
                "curvature_source": curvature_source,
            },
        )
        fitted = compiled.X @ final.beta
        residuals = compiled.y - fitted
        standardized = residuals / np.sqrt(
            np.maximum(np.diag(final.covariance), np.finfo(float).tiny)
        )
        return make_payload(
            parameters=natural_parameters,
            unconstrained_parameters=unconstrained_parameters,
            parameter_covariance=parameter_covariance,
            fitted_values=fitted,
            residuals=residuals,
            random_effects=np.zeros(0),
            objective=float(-final.log_likelihood),
            log_likelihood=float(final.log_likelihood),
            method=method,
            engine=self.name,
            convergence=convergence,
            diagnostic_data={
                "observations": {
                    "subject": [subject_levels[code] for code in subject_codes],
                    "visit": [visit_levels[code] for code in visit_codes],
                    "observed": compiled.y.tolist(),
                    "fitted": fitted.tolist(),
                    "residual": residuals.tolist(),
                    "standardized_residual": standardized.tolist(),
                },
                "estimated_marginal_means": emm_table or {},
                "contrasts": contrast_table or {},
            },
            extra={
                "parameter_names": list(compiled.fixed_names) + covariance_names,
                "unconstrained_parameter_names": list(compiled.fixed_names) + raw_names,
                "population_fitted_values": fitted,
                "fixed_design": compiled.X,
                "fixed_effect_covariance": final.beta_covariance,
                "fixed_effect_names": list(compiled.fixed_names),
                "residual_covariance": final.covariance,
                "covariance_structure": structure_name,
                "visit_levels": list(visit_levels),
                "visit_times": visit_times.tolist(),
                "visit_order_source": visit_axis.order_source,
                "declared_visit_levels": list(visit_axis.declared_levels),
                "visit_covariance": visit_matrix_hat,
                "subject_covariance": final.covariance,
                "covariance_parameter_covariance": theta_covariance,
                "profiled_fixed_effect_covariance": final.beta_covariance,
                "inference_fixed_effect_covariance": inference_covariance,
                "degrees_of_freedom_method": inference_label,
                "estimated_marginal_means": emm_table,
                "contrasts": contrast_table,
                "objective_convention": (
                    "negative exact normalized Gaussian restricted log likelihood"
                    if method == "REML"
                    else "negative exact normalized Gaussian marginal log likelihood"
                ),
                "likelihood_includes_data_constants": True,
                "reference_engine": True,
                "dense_subject_blocks": True,
                "kenward_roger_exact_available": False,
            },
        )


def fit_mmrm(data: Any, **options: Any) -> dict[str, Any]:
    """Fit a Gaussian MMRM with dense subject covariance blocks."""

    return MMRMBackend().fit(data, **options)


estimated_marginal_means = linear_inference
contrasts = linear_inference


__all__ = [
    "MMRMBackend",
    "contrasts",
    "estimated_marginal_means",
    "fit_mmrm",
    "linear_inference",
]
