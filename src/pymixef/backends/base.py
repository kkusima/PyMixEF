"""Shared contracts and numerical helpers for reference estimation backends.

Backends accept either a mapping or an object (including the compiler's
``DesignMatrices`` object).  If present, ``to_backend_data()`` is called first.
The common numeric fields are:

``response`` / ``y``
    One-dimensional response vector.
``fixed`` / ``X``
    Two-dimensional fixed-effect design matrix.
``fixed_names``
    Optional names matching the columns of ``X``.
``random_blocks``
    A sequence of mappings/objects.  Each block supplies ``matrix``/``Z`` and
    may supply row-level ``groups``/``group_codes``, ``group_levels``,
    ``term_names``, ``correlated``, and ``name``.  A row-level random design is
    expanded into one block of columns per group.  An already-expanded design
    can be marked with ``expanded=True``.
``residual_covariance``
    Optional explicit positive-definite covariance or correlation matrix.
    LMM treats it as a correlation template and estimates a common scale unless
    ``residual_covariance_fixed=True``.
``weights`` / ``offset`` / ``trials``
    Optional observation vectors used by applicable engines.

MMRM additionally consumes subject/group labels, visit labels, visit times, and
a covariance structure; those details are documented in ``mmrm.py``.

Every backend returns the backend-neutral mapping specified in ARCHITECTURE.md:
parameters, unconstrained_parameters, parameter_covariance, fitted_values,
residuals, random_effects, objective, log_likelihood, method, engine,
convergence, diagnostic_data, and extra.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg

try:  # The shared semantic error layer may be imported independently.
    from ..errors import EngineCompatibilityError, PyMixEFError  # type: ignore[attr-defined]
except (ImportError, AttributeError):

    class PyMixEFError(Exception):
        """Fallback used only while the shared error module is unavailable."""

    class EngineCompatibilityError(PyMixEFError):
        """Fallback compatibility error."""


class BackendError(PyMixEFError):
    """A stable, machine-readable backend input or numerical error."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "BACKEND-001",
        details: Mapping[str, Any] | None = None,
        suggested_engines: Sequence[str] = (),
    ) -> None:
        self.code = code
        self.message = message
        self.details = dict(details or {})
        self.suggested_engines = tuple(suggested_engines)
        super().__init__(f"{code}: {message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
            "suggested_engines": list(self.suggested_engines),
        }


class BackendInputError(BackendError):
    """The compiled numeric input is invalid."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message, code="BACKEND-INPUT-001", details=details)


class BackendUnsupportedError(EngineCompatibilityError):
    """The requested model is scientifically unsupported by this engine."""

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        suggested_engines: Sequence[str] = (),
    ) -> None:
        self.code = "ENGINE-UNSUPPORTED-001"
        self.message = message
        self.details = dict(details or {})
        self.suggested_engines = tuple(suggested_engines)
        try:
            super().__init__(
                message,
                code=self.code,
                details=self.details,
                suggested_engines=self.suggested_engines,
            )
        except TypeError:  # pragma: no cover - only the isolated fallback
            super().__init__(f"{self.code}: {message}")

    def to_dict(self) -> dict[str, Any]:
        parent = getattr(super(), "to_dict", None)
        if callable(parent):
            result = parent()
            result["suggested_engines"] = list(self.suggested_engines)
            return result
        return {
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
            "suggested_engines": list(self.suggested_engines),
        }


class BackendNumericalError(BackendError):
    """A numerical calculation failed before a valid payload could be made."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message, code="BACKEND-NUMERICAL-001", details=details)


@runtime_checkable
class Backend(Protocol):
    """Protocol implemented by all estimation engines."""

    name: str

    def fit(self, data: Any, **options: Any) -> dict[str, Any]:
        """Fit a compiled numeric model and return the shared payload."""


_MISSING = object()


def field(source: Any, *names: str, default: Any = _MISSING) -> Any:
    """Retrieve the first matching mapping key or object attribute."""

    for name in names:
        if isinstance(source, Mapping) and name in source:
            return source[name]
        if hasattr(source, name):
            return getattr(source, name)
    if default is _MISSING:
        joined = "/".join(names)
        raise BackendInputError(f"required backend field {joined!r} is missing")
    return default


def backend_mapping(data: Any) -> Any:
    """Resolve an optional ``to_backend_data`` conversion."""

    converter = getattr(data, "to_backend_data", None)
    if callable(converter):
        data = converter()
    return data


def _matrix(value: ArrayLike, name: str, *, rows: int | None = None) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=float)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2:
        raise BackendInputError(f"{name} must be a two-dimensional numeric matrix")
    if rows is not None and result.shape[0] != rows:
        raise BackendInputError(
            f"{name} has {result.shape[0]} rows; expected {rows}",
            details={"actual_rows": result.shape[0], "expected_rows": rows},
        )
    if np.any(~np.isfinite(result)):
        raise BackendInputError(f"{name} contains non-finite values")
    return result


def _vector(value: ArrayLike, name: str, *, length: int | None = None) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1:
        result = np.ravel(result)
    if length is not None and result.size != length:
        raise BackendInputError(
            f"{name} has length {result.size}; expected {length}",
            details={"actual_length": result.size, "expected_length": length},
        )
    if np.any(~np.isfinite(result)):
        raise BackendInputError(f"{name} contains non-finite values")
    return result


def factorize(values: ArrayLike) -> tuple[NDArray[np.int64], tuple[Any, ...]]:
    """Factorize values in first-observed order without a pandas dependency."""

    array = np.asarray(values)
    if array.ndim != 1:
        array = np.ravel(array)
    levels: list[Any] = []
    lookup: dict[Any, int] = {}
    codes = np.empty(array.size, dtype=np.int64)
    for index, raw in enumerate(array.tolist()):
        key = raw.item() if isinstance(raw, np.generic) else raw
        try:
            code = lookup[key]
        except KeyError:
            code = len(levels)
            lookup[key] = code
            levels.append(key)
        codes[index] = code
    return codes, tuple(levels)


@dataclass(slots=True)
class RandomBlockData:
    """One expanded random-effect design block."""

    name: str
    row_design: NDArray[np.float64]
    design: NDArray[np.float64]
    group_codes: NDArray[np.int64]
    group_levels: tuple[Any, ...]
    term_names: tuple[str, ...]
    correlated: bool
    terms_per_group: int

    @property
    def n_groups(self) -> int:
        return len(self.group_levels)

    @property
    def coefficient_count(self) -> int:
        return self.design.shape[1]

    def coefficient_labels(self) -> list[str]:
        labels: list[str] = []
        for level in self.group_levels:
            labels.extend(f"{self.name}[{level!s}].{term}" for term in self.term_names)
        return labels


def prepare_random_block(raw: Any, n_rows: int, index: int) -> RandomBlockData:
    """Validate and expand one compiler random block."""

    row_design = _matrix(
        field(raw, "matrix", "Z", "design"), f"random_blocks[{index}].Z", rows=n_rows
    )
    name = str(field(raw, "name", "group_name", default=f"random{index + 1}"))
    explicit_codes = field(raw, "group_codes", default=None)
    groups = (
        explicit_codes
        if explicit_codes is not None
        else field(raw, "groups", "group_labels", "group", default=None)
    )
    expanded = bool(field(raw, "expanded", "is_expanded", default=False))
    raw_term_names = field(raw, "term_names", "column_names", default=None)
    correlated = bool(field(raw, "correlated", default=True))

    if groups is None:
        codes = np.zeros(n_rows, dtype=np.int64)
        levels = tuple(field(raw, "group_levels", default=("all",)))
        if expanded:
            # Without row-level group codes, an expanded design is one covariance
            # block.  Its columns are separate terms.
            terms_per_group = row_design.shape[1]
            levels = ("all",)
        else:
            terms_per_group = row_design.shape[1]
    else:
        raw_groups = np.asarray(groups)
        if raw_groups.ndim != 1 or raw_groups.size != n_rows:
            raise BackendInputError(
                f"random block {name!r} group labels must have one entry per observation"
            )
        # Integer group_codes may arrive with a separate level vector.
        supplied_levels = field(raw, "group_levels", "levels", default=None)
        if explicit_codes is not None:
            codes = raw_groups.astype(np.int64)
            levels = (
                tuple(np.asarray(supplied_levels).tolist())
                if supplied_levels is not None
                else tuple(range(int(codes.max()) + 1))
            )
            if np.any(codes < 0) or np.any(codes >= len(levels)):
                raise BackendInputError(f"random block {name!r} contains invalid group codes")
        else:
            codes, levels = factorize(raw_groups)
        if expanded:
            if row_design.shape[1] % len(levels):
                raise BackendInputError(
                    f"expanded random block {name!r} columns are not divisible by group count"
                )
            terms_per_group = row_design.shape[1] // len(levels)
        else:
            terms_per_group = row_design.shape[1]

    if raw_term_names is None:
        term_names = tuple(
            "intercept" if terms_per_group == 1 and j == 0 else f"term{j + 1}"
            for j in range(terms_per_group)
        )
    else:
        term_names = tuple(str(item) for item in raw_term_names)
        if len(term_names) == row_design.shape[1] and expanded and len(levels) > 1:
            term_names = term_names[:terms_per_group]
        if len(term_names) != terms_per_group:
            raise BackendInputError(
                f"random block {name!r} has {len(term_names)} term names but "
                f"{terms_per_group} terms per group"
            )

    if expanded:
        design = row_design
    else:
        design = np.zeros((n_rows, len(levels) * terms_per_group), dtype=float)
        rows = np.arange(n_rows)
        for term in range(terms_per_group):
            design[rows, codes * terms_per_group + term] = row_design[:, term]

    return RandomBlockData(
        name=name,
        row_design=row_design,
        design=design,
        group_codes=codes,
        group_levels=levels,
        term_names=term_names,
        correlated=correlated,
        terms_per_group=terms_per_group,
    )


@dataclass(slots=True)
class CompiledData:
    """Normalized fields shared by LMM and GLMM."""

    source: Any
    y: NDArray[np.float64]
    X: NDArray[np.float64]
    fixed_names: tuple[str, ...]
    random_blocks: tuple[RandomBlockData, ...]
    offset: NDArray[np.float64]
    weights: NDArray[np.float64]
    trials: NDArray[np.float64] | None
    residual_covariance: Any
    residual_covariance_fixed: bool

    @property
    def n_obs(self) -> int:
        return self.y.size

    @property
    def n_fixed(self) -> int:
        return self.X.shape[1]

    @property
    def random_design(self) -> NDArray[np.float64]:
        if not self.random_blocks:
            return np.zeros((self.n_obs, 0), dtype=float)
        return np.concatenate([block.design for block in self.random_blocks], axis=1)


def prepare_data(data: Any, *, require_random: bool = False) -> CompiledData:
    """Normalize common compiler output for a numerical backend."""

    source = backend_mapping(data)
    y = _vector(field(source, "response", "y"), "response")
    X = _matrix(field(source, "fixed", "X"), "fixed", rows=y.size)
    raw_names = field(source, "fixed_names", "column_names", default=None)
    if raw_names is None:
        fixed_names = tuple(f"beta[{j}]" for j in range(X.shape[1]))
    else:
        fixed_names = tuple(str(item) for item in raw_names)
        if len(fixed_names) != X.shape[1]:
            raise BackendInputError("fixed_names length does not match fixed design columns")

    raw_blocks = field(source, "random_blocks", "random", default=())
    if raw_blocks is None:
        raw_blocks = ()
    blocks = tuple(prepare_random_block(item, y.size, j) for j, item in enumerate(raw_blocks))
    if require_random and not blocks:
        raise BackendInputError("this engine requires at least one random-effects block")

    offset_value = field(source, "offset", default=None)
    offset = _vector(
        np.zeros(y.size) if offset_value is None else offset_value,
        "offset",
        length=y.size,
    )
    weights_value = field(source, "weights", default=None)
    weights = _vector(
        np.ones(y.size) if weights_value is None else weights_value,
        "weights",
        length=y.size,
    )
    if np.any(weights <= 0):
        raise BackendInputError("weights must be strictly positive")
    trials_raw = field(source, "trials", "binomial_trials", default=None)
    trials = None if trials_raw is None else _vector(trials_raw, "trials", length=y.size)

    return CompiledData(
        source=source,
        y=y,
        X=X,
        fixed_names=fixed_names,
        random_blocks=blocks,
        offset=offset,
        weights=weights,
        trials=trials,
        residual_covariance=field(source, "residual_covariance", "residual", "R", default=None),
        residual_covariance_fixed=bool(
            field(source, "residual_covariance_fixed", "R_fixed", default=False)
        ),
    )


@dataclass(slots=True)
class CovarianceParameterization:
    """Positive-definite random covariance parameterization."""

    block: RandomBlockData

    @property
    def size(self) -> int:
        q = self.block.terms_per_group
        return q if not self.block.correlated else q * (q + 1) // 2

    def initial(self, scale: float) -> NDArray[np.float64]:
        q = self.block.terms_per_group
        if not self.block.correlated:
            return np.full(q, np.log(max(scale, 1e-3)))
        result: list[float] = []
        for row in range(q):
            for col in range(row + 1):
                result.append(np.log(max(scale, 1e-3)) if row == col else 0.0)
        return np.asarray(result)

    def matrix(self, theta: ArrayLike) -> NDArray[np.float64]:
        theta = np.asarray(theta, dtype=float)
        q = self.block.terms_per_group
        if theta.size != self.size:
            raise ValueError("wrong covariance parameter vector length")
        if not self.block.correlated:
            return np.diag(np.exp(2 * theta))
        chol = np.zeros((q, q), dtype=float)
        cursor = 0
        for row in range(q):
            for col in range(row + 1):
                chol[row, col] = np.exp(theta[cursor]) if row == col else theta[cursor]
                cursor += 1
        return chol @ chol.T

    def expanded_matrix(self, theta: ArrayLike) -> NDArray[np.float64]:
        return np.kron(np.eye(self.block.n_groups), self.matrix(theta))

    def names_and_values(self, theta: ArrayLike) -> dict[str, float]:
        covariance = self.matrix(theta)
        values: dict[str, float] = {}
        q = self.block.terms_per_group
        std = np.sqrt(np.diag(covariance))
        for j, term in enumerate(self.block.term_names):
            values[f"sd({self.block.name}:{term})"] = float(std[j])
        if self.block.correlated:
            for row in range(1, q):
                for col in range(row):
                    values[
                        f"corr({self.block.name}:{self.block.term_names[row]},"
                        f"{self.block.term_names[col]})"
                    ] = float(covariance[row, col] / (std[row] * std[col]))
        return values

    def unconstrained_names(self) -> list[str]:
        q = self.block.terms_per_group
        if not self.block.correlated:
            return [f"log_sd({self.block.name}:{term})" for term in self.block.term_names]
        result: list[str] = []
        for row in range(q):
            for col in range(row + 1):
                if row == col:
                    result.append(f"log_chol({self.block.name}:{self.block.term_names[row]})")
                else:
                    result.append(
                        f"chol({self.block.name}:{self.block.term_names[row]},"
                        f"{self.block.term_names[col]})"
                    )
        return result


def covariance_slices(
    blocks: Sequence[RandomBlockData],
) -> tuple[tuple[CovarianceParameterization, slice], ...]:
    """Build covariance parameterizations and their theta slices."""

    output: list[tuple[CovarianceParameterization, slice]] = []
    cursor = 0
    for block in blocks:
        parameterization = CovarianceParameterization(block)
        output.append((parameterization, slice(cursor, cursor + parameterization.size)))
        cursor += parameterization.size
    return tuple(output)


def random_covariance(
    parameterizations: Sequence[tuple[CovarianceParameterization, slice]],
    theta: ArrayLike,
) -> NDArray[np.float64]:
    """Assemble block-diagonal covariance for all expanded coefficients."""

    theta = np.asarray(theta, dtype=float)
    matrices = [item.expanded_matrix(theta[section]) for item, section in parameterizations]
    if not matrices:
        return np.zeros((0, 0), dtype=float)
    return linalg.block_diag(*matrices)


def safe_cholesky(
    matrix: ArrayLike, *, jitter_scale: float = 1e-10, max_tries: int = 5
) -> tuple[NDArray[np.float64], float]:
    """Cholesky factor a symmetric matrix, adding only reported tiny jitter."""

    matrix = np.asarray(matrix, dtype=float)
    matrix = (matrix + matrix.T) / 2
    scale = max(float(np.max(np.abs(np.diag(matrix)))), 1.0)
    jitter = 0.0
    for attempt in range(max_tries):
        try:
            return linalg.cholesky(matrix + jitter * np.eye(matrix.shape[0]), lower=True), jitter
        except linalg.LinAlgError:
            jitter = jitter_scale * scale * (10**attempt)
    raise linalg.LinAlgError("matrix is not positive definite after bounded jitter")


def cho_solve(cholesky: NDArray[np.float64], value: ArrayLike) -> NDArray[np.float64]:
    """Solve against a lower Cholesky factor."""

    return linalg.cho_solve((cholesky, True), np.asarray(value, dtype=float), check_finite=False)


def logdet_from_cholesky(cholesky: NDArray[np.float64]) -> float:
    """Return the log determinant represented by a lower Cholesky factor.

    ``cholesky`` is assumed to have a positive diagonal; this helper does not
    validate that the factor came from a positive-definite matrix.
    """

    return float(2 * np.log(np.diag(cholesky)).sum())


def finite_hessian(
    function: Any,
    point: ArrayLike,
    *,
    relative_step: float = 2e-4,
) -> NDArray[np.float64]:
    """Central finite-difference Hessian for small reference problems."""

    x = np.asarray(point, dtype=float)
    n = x.size
    h = relative_step * np.maximum(1.0, np.abs(x))
    result = np.empty((n, n), dtype=float)
    f0 = float(function(x))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]
        result[i, i] = (float(function(x + ei)) - 2 * f0 + float(function(x - ei))) / h[i] ** 2
        for j in range(i):
            ej = np.zeros(n)
            ej[j] = h[j]
            value = (
                float(function(x + ei + ej))
                - float(function(x + ei - ej))
                - float(function(x - ei + ej))
                + float(function(x - ei - ej))
            ) / (4 * h[i] * h[j])
            result[i, j] = result[j, i] = value
    return (result + result.T) / 2


def finite_gradient(
    function: Any,
    point: ArrayLike,
    *,
    relative_step: float = 1e-6,
) -> NDArray[np.float64]:
    """Central finite-difference gradient used for termination verification."""

    x = np.asarray(point, dtype=float)
    step = relative_step * np.maximum(1.0, np.abs(x))
    result = np.empty(x.size)
    for position in range(x.size):
        delta = np.zeros(x.size)
        delta[position] = step[position]
        result[position] = (float(function(x + delta)) - float(function(x - delta))) / (
            2 * step[position]
        )
    return result


def select_optimizer_result(
    refined: Any,
    rescue: Any,
    *,
    objective_tolerance: float,
) -> Any:
    """Prefer successful finite termination before comparing objective values.

    A line-search refinement can improve an objective by floating-point noise
    while still reporting an abnormal termination.  In that case, retaining a
    successfully terminated derivative-free rescue is more informative than
    replacing it with the failed refinement.
    """

    refined_objective = float(getattr(refined, "fun", np.nan))
    rescue_objective = float(getattr(rescue, "fun", np.nan))
    refined_finite = bool(np.isfinite(refined_objective))
    rescue_finite = bool(np.isfinite(rescue_objective))
    refined_converged = bool(getattr(refined, "success", False) and refined_finite)
    rescue_converged = bool(getattr(rescue, "success", False) and rescue_finite)

    if refined_converged != rescue_converged:
        converged = refined if refined_converged else rescue
        failed = rescue if refined_converged else refined
        converged_objective = refined_objective if refined_converged else rescue_objective
        failed_objective = rescue_objective if refined_converged else refined_objective
        if np.isfinite(failed_objective) and (
            failed_objective + objective_tolerance < converged_objective
        ):
            return failed
        return converged
    if refined_finite != rescue_finite:
        return refined if refined_finite else rescue
    if refined_finite and refined_objective <= rescue_objective + objective_tolerance:
        return refined
    return rescue


def optimizer_covariance(
    function: Any,
    point: ArrayLike,
    result: Any,
    *,
    compute_hessian: bool,
    finite_difference_limit: int,
) -> tuple[NDArray[np.float64], bool | None, str]:
    """Return auditable optimizer-scale covariance at the selected optimum."""

    location = np.asarray(point, dtype=float)
    if location.ndim != 1:
        raise ValueError("An optimizer point must be one-dimensional.")
    if location.size == 0:
        return np.zeros((0, 0)), True, "not-required"

    def observed_covariance(source: str) -> tuple[NDArray[np.float64], bool, str]:
        try:
            hessian = finite_hessian(function, location)
            covariance, positive = covariance_from_hessian(hessian)
        except (ValueError, FloatingPointError, linalg.LinAlgError) as exc:
            raise BackendNumericalError(
                "could not calculate finite covariance at the selected optimizer result",
                details={"curvature_source": source, "parameters": int(location.size)},
            ) from exc
        return covariance, positive, source

    if compute_hessian and location.size <= finite_difference_limit:
        return observed_covariance("observed-finite-difference")

    inverse = getattr(result, "hess_inv", None)
    try:
        dense_inverse = inverse.todense() if hasattr(inverse, "todense") else inverse
        covariance = np.asarray(dense_inverse, dtype=float)
        valid_inverse = bool(
            covariance.shape == (location.size, location.size)
            and np.all(np.isfinite(covariance))
            and np.allclose(covariance, covariance.T, rtol=1e-8, atol=1e-10)
        )
    except (TypeError, ValueError):
        valid_inverse = False
        covariance = np.empty((0, 0))
    if valid_inverse:
        return (covariance + covariance.T) / 2.0, None, "optimizer-inverse-hessian"

    if location.size <= finite_difference_limit:
        return observed_covariance("observed-finite-difference-fallback")

    raise BackendNumericalError(
        "the selected optimizer result did not provide a valid inverse Hessian",
        details={
            "parameters": int(location.size),
            "finite_difference_limit": int(finite_difference_limit),
        },
    )


def covariance_from_hessian(hessian: ArrayLike) -> tuple[NDArray[np.float64], bool]:
    """Return a symmetric generalized inverse and definiteness flag."""

    hessian = np.asarray(hessian, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh((hessian + hessian.T) / 2)
    threshold = max(np.max(np.abs(eigenvalues)), 1.0) * 1e-9
    positive = bool(np.all(eigenvalues > threshold))
    inverse_values = np.where(eigenvalues > threshold, 1.0 / eigenvalues, 0.0)
    return (eigenvectors * inverse_values) @ eigenvectors.T, positive


def convergence_mapping(
    *,
    success: bool,
    message: str,
    iterations: int,
    function_evaluations: int,
    gradient: ArrayLike | None,
    hessian_positive_definite: bool | None,
    singular: bool,
    boundary_parameters: Sequence[str] = (),
    warnings: Sequence[Mapping[str, Any]] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a rich convergence mapping accepted by the public result layer."""

    warning_records = [dict(item) for item in warnings]
    warning_codes = {str(item.get("code", "")) for item in warning_records}
    if gradient is None:
        gradient_norm = None
    else:
        gradient_array = np.asarray(gradient, dtype=float)
        gradient_norm = 0.0 if gradient_array.size == 0 else float(np.max(np.abs(gradient_array)))
    if (
        gradient_norm is not None
        and (not np.isfinite(gradient_norm) or gradient_norm > 1e-4)
        and "NUM-GRADIENT-LARGE-001" not in warning_codes
    ):
        warning_records.append(
            {
                "code": "NUM-GRADIENT-LARGE-001",
                "severity": "review",
                "message": "The scaled gradient remains larger than its convergence tolerance.",
                "component": "optimization",
                "remediation": "Rescale parameters, tighten controls, and compare "
                "independent starts or engines.",
                "details": {"scaled_gradient_inf_norm": gradient_norm},
            }
        )
    if hessian_positive_definite is False and "NUM-HESSIAN-INDEFINITE-001" not in warning_codes:
        warning_records.append(
            {
                "code": "NUM-HESSIAN-INDEFINITE-001",
                "severity": "review",
                "message": "The observed Hessian is not positive definite.",
                "component": "inference",
                "remediation": "Do not use naive Wald uncertainty; inspect gradients, "
                "profiles, and alternative starts.",
            }
        )
    if singular and "COV-SINGULAR-001" not in warning_codes:
        warning_records.append(
            {
                "code": "COV-SINGULAR-001",
                "severity": "review",
                "message": "An estimated covariance block is singular or near-singular.",
                "component": "covariance",
                "remediation": "Review the singularity report and random-effect design.",
            }
        )
    if boundary_parameters and "COV-BOUNDARY-001" not in warning_codes:
        warning_records.append(
            {
                "code": "COV-BOUNDARY-001",
                "severity": "review",
                "message": "One or more variance parameters are on a numerical boundary.",
                "component": "covariance",
                "details": {"parameters": list(boundary_parameters)},
            }
        )
    numerical_suspect = bool(
        (gradient_norm is not None and (not np.isfinite(gradient_norm) or gradient_norm > 1e-4))
        or hessian_positive_definite is False
    )
    warning_state = (
        singular or bool(boundary_parameters) or bool(warning_records) or numerical_suspect
    )
    status = "failed" if not success else ("warning" if warning_state else "converged")
    result: dict[str, Any] = {
        "status": status,
        "optimizer_terminated": bool(success),
        "message": str(message),
        "iterations": int(iterations),
        "function_evaluations": int(function_evaluations),
        "scaled_gradient_inf_norm": gradient_norm,
        "gradient_inf_norm": gradient_norm,
        "hessian_positive_definite": hessian_positive_definite,
        "singular": bool(singular),
        "boundary_parameters": list(boundary_parameters),
        "warnings": warning_records,
    }
    result.update(dict(extra or {}))
    return result


_PAYLOAD_KEYS = (
    "parameters",
    "unconstrained_parameters",
    "parameter_covariance",
    "fitted_values",
    "residuals",
    "random_effects",
    "objective",
    "log_likelihood",
    "method",
    "engine",
    "convergence",
    "diagnostic_data",
    "extra",
)


def make_payload(**values: Any) -> dict[str, Any]:
    """Validate and order the shared backend payload."""

    missing = [key for key in _PAYLOAD_KEYS if key not in values]
    if missing:
        raise BackendNumericalError(
            f"backend implementation omitted payload fields: {', '.join(missing)}"
        )
    return validate_payload({key: values[key] for key in _PAYLOAD_KEYS})


def validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the backend-neutral result contract and return canonical key order.

    The validator is shared by built-in and third-party backends. It checks
    presence, numeric finiteness, observation alignment, covariance shape,
    convergence state, and container types without imposing an
    engine-specific parameterization.
    """

    missing = [key for key in _PAYLOAD_KEYS if key not in payload]
    if missing:
        raise BackendNumericalError(
            f"backend implementation omitted payload fields: {', '.join(missing)}"
        )
    parameters = payload["parameters"]
    unconstrained = payload["unconstrained_parameters"]
    if not isinstance(parameters, Mapping) or not parameters:
        raise BackendNumericalError("backend parameters must be a non-empty mapping")
    if not isinstance(unconstrained, Mapping):
        raise BackendNumericalError("backend unconstrained_parameters must be a mapping")
    for label, values in (
        ("parameters", parameters.values()),
        ("unconstrained_parameters", unconstrained.values()),
    ):
        try:
            numeric = np.asarray([float(value) for value in values], dtype=float)
        except (TypeError, ValueError) as error:
            raise BackendNumericalError(f"backend {label} must be numeric") from error
        if np.any(~np.isfinite(numeric)):
            raise BackendNumericalError(f"backend {label} contains non-finite values")

    try:
        fitted = np.asarray(payload["fitted_values"], dtype=float)
        residuals = np.asarray(payload["residuals"], dtype=float)
    except (TypeError, ValueError) as error:
        raise BackendNumericalError(
            "backend fitted_values and residuals must be numeric"
        ) from error
    if fitted.ndim != 1 or residuals.ndim != 1 or fitted.shape != residuals.shape:
        raise BackendNumericalError(
            "backend fitted_values and residuals must be aligned one-dimensional arrays"
        )
    if np.any(~np.isfinite(fitted)) or np.any(~np.isfinite(residuals)):
        raise BackendNumericalError("backend fitted_values or residuals contain non-finite values")

    try:
        objective = float(payload["objective"])
    except (TypeError, ValueError) as error:
        raise BackendNumericalError("backend objective must be numeric") from error
    log_likelihood = payload["log_likelihood"]
    if not np.isfinite(objective):
        raise BackendNumericalError("backend objective must be finite")
    if log_likelihood is not None:
        try:
            numeric_log_likelihood = float(log_likelihood)
        except (TypeError, ValueError) as error:
            raise BackendNumericalError("backend log_likelihood must be numeric or None") from error
        if not np.isfinite(numeric_log_likelihood):
            raise BackendNumericalError("backend log_likelihood must be finite or None")

    covariance = payload["parameter_covariance"]
    if covariance is not None:
        try:
            covariance_array = np.asarray(covariance, dtype=float)
        except (TypeError, ValueError) as error:
            raise BackendNumericalError("backend parameter_covariance must be numeric") from error
        if covariance_array.ndim != 2 or covariance_array.shape[0] != covariance_array.shape[1]:
            raise BackendNumericalError("backend parameter_covariance must be square")
        if np.any(~np.isfinite(covariance_array)):
            raise BackendNumericalError("backend parameter_covariance contains non-finite values")
        if not np.allclose(covariance_array, covariance_array.T, rtol=1e-8, atol=1e-10):
            raise BackendNumericalError("backend parameter_covariance must be symmetric")

    convergence = payload["convergence"]
    if not isinstance(convergence, Mapping):
        raise BackendNumericalError("backend convergence must be a mapping")
    if convergence.get("status") not in {"converged", "warning", "failed"}:
        raise BackendNumericalError(
            "backend convergence status must be converged, warning, or failed"
        )
    if (
        not isinstance(payload["method"], str)
        or not payload["method"].strip()
        or not isinstance(payload["engine"], str)
        or not payload["engine"].strip()
    ):
        raise BackendNumericalError("backend method and engine must be non-empty")
    if not isinstance(payload["diagnostic_data"], Mapping):
        raise BackendNumericalError("backend diagnostic_data must be a mapping")
    if not isinstance(payload["extra"], Mapping):
        raise BackendNumericalError("backend extra must be a mapping")

    return {key: payload[key] for key in _PAYLOAD_KEYS}


__all__ = [
    "Backend",
    "BackendError",
    "BackendInputError",
    "BackendNumericalError",
    "BackendUnsupportedError",
    "CompiledData",
    "CovarianceParameterization",
    "RandomBlockData",
    "backend_mapping",
    "cho_solve",
    "convergence_mapping",
    "covariance_from_hessian",
    "covariance_slices",
    "factorize",
    "field",
    "finite_gradient",
    "finite_hessian",
    "logdet_from_cholesky",
    "make_payload",
    "optimizer_covariance",
    "prepare_data",
    "prepare_random_block",
    "random_covariance",
    "safe_cholesky",
    "select_optimizer_result",
    "validate_payload",
]
