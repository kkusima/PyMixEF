"""Positive-definite covariance structures and diagnostics.

All estimable structures accept unconstrained optimizer parameters and construct
valid covariance matrices throughout optimization.  Parameters are ordered
deterministically and documented by :meth:`parameter_names`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

from .errors import CovarianceError
from .plugins import COVARIANCE_REGISTRY

FloatArray = NDArray[np.float64]


def _parameters(values: ArrayLike, expected: int) -> FloatArray:
    result = np.asarray(values, dtype=float).reshape(-1)
    if result.size != expected:
        raise CovarianceError(
            f"Expected {expected} unconstrained covariance parameters; got {result.size}.",
            code="COV-PARAMETER-COUNT-001",
            details={"expected": expected, "actual": int(result.size)},
        )
    if not np.all(np.isfinite(result)):
        raise CovarianceError("Covariance parameters must be finite.")
    return result


def _dimension(
    configured: int | None,
    size: int | None,
    index: ArrayLike | None,
    *,
    inferred: int | None = None,
) -> int:
    if configured is not None and size is not None and configured != size:
        raise CovarianceError("Configured and requested covariance dimensions differ.")
    result = configured or size
    if result is None and index is not None:
        result = int(np.asarray(index).shape[0])
    result = result or inferred
    if result is None or result < 1:
        raise CovarianceError(
            "Covariance dimension is required.",
            code="COV-DIMENSION-001",
        )
    return result


def validate_covariance(
    matrix: ArrayLike,
    *,
    positive_semidefinite: bool = False,
    tolerance: float = 1e-10,
) -> CovarianceValidation:
    """Validate symmetry and definiteness and return numerical diagnostics."""

    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise CovarianceError(
            "Covariance matrix must be square.",
            code="COV-SHAPE-001",
            details={"shape": value.shape},
        )
    if not np.all(np.isfinite(value)):
        raise CovarianceError("Covariance matrix contains non-finite values.")
    symmetric = bool(np.allclose(value, value.T, rtol=0.0, atol=tolerance))
    if not symmetric:
        raise CovarianceError(
            "Covariance matrix is not symmetric.",
            code="COV-SYMMETRY-001",
        )
    eigenvalues = np.linalg.eigvalsh((value + value.T) / 2.0)
    threshold = tolerance * max(1.0, float(np.max(np.abs(eigenvalues))))
    if positive_semidefinite:
        valid = bool(np.min(eigenvalues) >= -threshold)
    else:
        try:
            np.linalg.cholesky((value + value.T) / 2.0)
        except np.linalg.LinAlgError:
            valid = False
        else:
            valid = True
    if not valid:
        raise CovarianceError(
            "Covariance matrix is not positive definite.",
            code="COV-POSDEF-001",
            details={"eigenvalues": eigenvalues.tolist()},
        )
    rank = int(np.sum(eigenvalues > threshold))
    ratio = float(eigenvalues[0] / eigenvalues[-1]) if eigenvalues[-1] > 0 else 0.0
    return CovarianceValidation(
        dimension=value.shape[0],
        symmetric=symmetric,
        positive_definite=bool(eigenvalues[0] > threshold),
        effective_rank=rank,
        eigenvalues=tuple(float(item) for item in eigenvalues),
        eigenvalue_ratio=ratio,
        near_singular=ratio < 1e-8,
    )


def _stabilize_correlation(matrix: FloatArray) -> FloatArray:
    """Preserve structure while avoiding floating-point boundary singularity."""

    value = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0
    minimum = float(np.linalg.eigvalsh(value)[0])
    floor = 32.0 * np.finfo(float).eps * max(1, value.shape[0])
    if minimum < floor:
        value = value + np.eye(value.shape[0]) * (floor - minimum)
    scales = np.sqrt(np.diag(value))
    return value / scales[:, None] / scales[None, :]


@dataclass(frozen=True, slots=True)
class CovarianceValidation:
    """Numerical validation report for a covariance matrix."""

    dimension: int
    symmetric: bool
    positive_definite: bool
    effective_rank: int
    eigenvalues: tuple[float, ...]
    eigenvalue_ratio: float
    near_singular: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "symmetric": self.symmetric,
            "positive_definite": self.positive_definite,
            "effective_rank": self.effective_rank,
            "eigenvalues": list(self.eigenvalues),
            "eigenvalue_ratio": self.eigenvalue_ratio,
            "near_singular": self.near_singular,
        }


class CovarianceStructure:
    """Base class for covariance declarations and estimable kernels."""

    name = "covariance"

    def __init__(
        self,
        dimension: int | None = None,
        *,
        index: str | None = None,
        group: str | None = None,
    ) -> None:
        if dimension is not None and dimension < 1:
            raise CovarianceError("Covariance dimension must be positive.")
        self.dimension = dimension
        self.index = index
        self.group = group

    def parameter_count(self, size: int | None = None) -> int:
        """Number of unconstrained parameters for a matrix dimension."""

        raise NotImplementedError

    def parameter_names(self, size: int | None = None) -> tuple[str, ...]:
        """Names matching the unconstrained parameter ordering."""

        return tuple(f"{self.name}_{i}" for i in range(self.parameter_count(size)))

    def covariance(
        self,
        parameters: ArrayLike,
        *,
        size: int | None = None,
        index: ArrayLike | None = None,
        coordinates: ArrayLike | None = None,
    ) -> FloatArray:
        """Construct the covariance matrix."""

        raise NotImplementedError

    def matrix(self, parameters: ArrayLike, **context: Any) -> FloatArray:
        """Alias for :meth:`covariance`."""

        return self.covariance(parameters, **context)

    __call__ = matrix

    def validate(self, matrix: ArrayLike, **_: Any) -> CovarianceValidation:
        """Return definiteness diagnostics or raise :class:`CovarianceError`."""

        return validate_covariance(matrix)

    def derivatives(self, parameters: ArrayLike, **context: Any) -> FloatArray:
        """Return central finite-difference derivatives ``(p, n, n)``."""

        theta = np.asarray(parameters, dtype=float).reshape(-1)
        base = self.covariance(theta, **context)
        result = np.empty((theta.size, *base.shape), dtype=float)
        step = np.cbrt(np.finfo(float).eps)
        for position in range(theta.size):
            delta = step * max(1.0, abs(theta[position]))
            upper = theta.copy()
            lower = theta.copy()
            upper[position] += delta
            lower[position] -= delta
            result[position] = (
                self.covariance(upper, **context) - self.covariance(lower, **context)
            ) / (2.0 * delta)
        return result

    def simulate(
        self,
        parameters: ArrayLike,
        *,
        rng: np.random.Generator | int | None = None,
        draws: int = 1,
        **context: Any,
    ) -> FloatArray:
        """Draw zero-mean multivariate normal realizations."""

        if draws < 1:
            raise CovarianceError("draws must be positive.")
        generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        matrix = self.covariance(parameters, **context)
        samples = generator.multivariate_normal(np.zeros(matrix.shape[0]), matrix, size=draws)
        return samples[0] if draws == 1 else samples

    def to_dict(self) -> dict[str, Any]:
        """Serialize the declaration (not estimated parameter values)."""

        return {
            "structure": self.name,
            "dimension": self.dimension,
            "index": self.index,
            "group": self.group,
        }


class Diagonal(CovarianceStructure):
    """Independent components parameterized by log standard deviations."""

    name = "diagonal"

    def parameter_count(self, size: int | None = None) -> int:
        return _dimension(self.dimension, size, None)

    def parameter_names(self, size: int | None = None) -> tuple[str, ...]:
        return tuple(f"log_sd_{i + 1}" for i in range(self.parameter_count(size)))

    def covariance(self, parameters: ArrayLike, *, size: int | None = None, **_: Any) -> FloatArray:
        n = _dimension(self.dimension, size, None, inferred=np.asarray(parameters).size)
        theta = _parameters(parameters, n)
        return np.diag(np.exp(2.0 * theta))


class Unstructured(CovarianceStructure):
    """Unstructured SPD covariance parameterized by a Cholesky factor."""

    name = "unstructured"

    def parameter_count(self, size: int | None = None) -> int:
        n = _dimension(self.dimension, size, None)
        return n * (n + 1) // 2

    def parameter_names(self, size: int | None = None) -> tuple[str, ...]:
        n = _dimension(self.dimension, size, None)
        return tuple(
            ("log_chol" if row == column else "chol") + f"_{row + 1}_{column + 1}"
            for row in range(n)
            for column in range(row + 1)
        )

    def covariance(self, parameters: ArrayLike, *, size: int | None = None, **_: Any) -> FloatArray:
        raw = np.asarray(parameters)
        inferred = int((np.sqrt(8 * raw.size + 1) - 1) / 2)
        n = _dimension(self.dimension, size, None, inferred=inferred)
        theta = _parameters(parameters, n * (n + 1) // 2)
        lower = np.zeros((n, n))
        cursor = 0
        for row in range(n):
            for column in range(row + 1):
                lower[row, column] = np.exp(theta[cursor]) if row == column else theta[cursor]
                cursor += 1
        return lower @ lower.T


class CompoundSymmetry(CovarianceStructure):
    """Homogeneous exchangeable covariance with a valid dimension-aware range."""

    name = "compound-symmetry"

    def parameter_count(self, size: int | None = None) -> int:
        _dimension(self.dimension, size, None)
        return 2

    def parameter_names(self, size: int | None = None) -> tuple[str, ...]:
        return ("log_sd", "correlation_unconstrained")

    def covariance(self, parameters: ArrayLike, *, size: int | None = None, **_: Any) -> FloatArray:
        n = _dimension(self.dimension, size, None)
        theta = _parameters(parameters, 2)
        lower = -1.0 / (n - 1) if n > 1 else 0.0
        rho = lower + (1.0 - lower) * expit(theta[1])
        correlation = np.full((n, n), rho)
        np.fill_diagonal(correlation, 1.0)
        correlation = _stabilize_correlation(correlation)
        return np.exp(2.0 * theta[0]) * correlation


def _ordered_distances(index: ArrayLike | None, n: int) -> FloatArray:
    values = (
        np.arange(n, dtype=float) if index is None else np.asarray(index, dtype=float).reshape(-1)
    )
    if values.size != n or not np.all(np.isfinite(values)):
        raise CovarianceError("The covariance index must contain one finite value per row.")
    return np.abs(values[:, None] - values[None, :])


class AR1(CovarianceStructure):
    """Homogeneous first-order autoregressive covariance."""

    name = "ar1"

    def parameter_count(self, size: int | None = None) -> int:
        _dimension(self.dimension, size, None)
        return 2

    def parameter_names(self, size: int | None = None) -> tuple[str, ...]:
        return ("log_sd", "correlation_unconstrained")

    def covariance(
        self,
        parameters: ArrayLike,
        *,
        size: int | None = None,
        index: ArrayLike | None = None,
        **_: Any,
    ) -> FloatArray:
        n = _dimension(self.dimension, size, index)
        theta = _parameters(parameters, 2)
        rho = float(np.tanh(theta[1]))
        distances = _ordered_distances(index, n)
        if rho < 0 and not np.allclose(distances, np.round(distances)):
            raise CovarianceError(
                "Negative AR(1) correlation requires integer index spacing.",
                code="COV-AR1-SPACING-001",
            )
        correlation = np.sign(rho) ** np.round(distances) * abs(rho) ** distances
        correlation = _stabilize_correlation(correlation)
        return np.exp(2.0 * theta[0]) * correlation


class HeterogeneousAR1(CovarianceStructure):
    """AR(1) correlation with one log standard deviation per position."""

    name = "heterogeneous-ar1"

    def parameter_count(self, size: int | None = None) -> int:
        return _dimension(self.dimension, size, None) + 1

    def parameter_names(self, size: int | None = None) -> tuple[str, ...]:
        n = _dimension(self.dimension, size, None)
        return tuple(f"log_sd_{i + 1}" for i in range(n)) + ("correlation_unconstrained",)

    def covariance(
        self,
        parameters: ArrayLike,
        *,
        size: int | None = None,
        index: ArrayLike | None = None,
        **_: Any,
    ) -> FloatArray:
        inferred = np.asarray(parameters).size - 1
        n = _dimension(self.dimension, size, index, inferred=inferred)
        theta = _parameters(parameters, n + 1)
        rho = float(np.tanh(theta[-1]))
        distances = _ordered_distances(index, n)
        if rho < 0 and not np.allclose(distances, np.round(distances)):
            raise CovarianceError(
                "Negative heterogeneous AR(1) correlation requires integer spacing.",
                code="COV-AR1-SPACING-001",
            )
        correlation = np.sign(rho) ** np.round(distances) * abs(rho) ** distances
        correlation = _stabilize_correlation(correlation)
        scales = np.exp(theta[:-1])
        return scales[:, None] * correlation * scales[None, :]


def _pacf_to_acf(pacf: FloatArray) -> FloatArray:
    """Durbin-Levinson recursion from partial to ordinary autocorrelations."""

    order = pacf.size
    acf = np.ones(order + 1)
    previous = np.zeros(order + 1)
    for current_order in range(1, order + 1):
        current = np.zeros(order + 1)
        current[current_order] = pacf[current_order - 1]
        if current_order > 1:
            for lag in range(1, current_order):
                current[lag] = (
                    previous[lag] - pacf[current_order - 1] * previous[current_order - lag]
                )
        acf[current_order] = sum(
            current[lag] * acf[current_order - lag] for lag in range(1, current_order + 1)
        )
        previous = current
    return acf


def _toeplitz_correlation(pacf: FloatArray) -> FloatArray:
    acf = _pacf_to_acf(pacf)
    positions = np.arange(acf.size)
    return _stabilize_correlation(acf[np.abs(positions[:, None] - positions[None, :])])


class Toeplitz(CovarianceStructure):
    """Stationary Toeplitz covariance using unconstrained partial correlations."""

    name = "toeplitz"

    def parameter_count(self, size: int | None = None) -> int:
        return _dimension(self.dimension, size, None)

    def parameter_names(self, size: int | None = None) -> tuple[str, ...]:
        n = _dimension(self.dimension, size, None)
        return ("log_sd",) + tuple(f"partial_correlation_{lag}" for lag in range(1, n))

    def covariance(self, parameters: ArrayLike, *, size: int | None = None, **_: Any) -> FloatArray:
        inferred = np.asarray(parameters).size
        n = _dimension(self.dimension, size, None, inferred=inferred)
        theta = _parameters(parameters, n)
        correlation = _toeplitz_correlation(np.tanh(theta[1:]))
        return np.exp(2.0 * theta[0]) * correlation


class HeterogeneousToeplitz(CovarianceStructure):
    """Toeplitz correlation with position-specific standard deviations."""

    name = "heterogeneous-toeplitz"

    def parameter_count(self, size: int | None = None) -> int:
        n = _dimension(self.dimension, size, None)
        return 2 * n - 1

    def parameter_names(self, size: int | None = None) -> tuple[str, ...]:
        n = _dimension(self.dimension, size, None)
        return tuple(f"log_sd_{i + 1}" for i in range(n)) + tuple(
            f"partial_correlation_{lag}" for lag in range(1, n)
        )

    def covariance(self, parameters: ArrayLike, *, size: int | None = None, **_: Any) -> FloatArray:
        raw_size = np.asarray(parameters).size
        inferred = (raw_size + 1) // 2
        n = _dimension(self.dimension, size, None, inferred=inferred)
        theta = _parameters(parameters, 2 * n - 1)
        scales = np.exp(theta[:n])
        correlation = _toeplitz_correlation(np.tanh(theta[n:]))
        return scales[:, None] * correlation * scales[None, :]


class AnteDependence(CovarianceStructure):
    """First-order ante-dependence with stable innovation standard deviations."""

    name = "ante-dependence"

    def parameter_count(self, size: int | None = None) -> int:
        n = _dimension(self.dimension, size, None)
        return 2 * n - 1

    def parameter_names(self, size: int | None = None) -> tuple[str, ...]:
        n = _dimension(self.dimension, size, None)
        return tuple(f"log_innovation_sd_{i + 1}" for i in range(n)) + tuple(
            f"dependence_{i + 2}_on_{i + 1}" for i in range(n - 1)
        )

    def covariance(self, parameters: ArrayLike, *, size: int | None = None, **_: Any) -> FloatArray:
        raw_size = np.asarray(parameters).size
        inferred = (raw_size + 1) // 2
        n = _dimension(self.dimension, size, None, inferred=inferred)
        theta = _parameters(parameters, 2 * n - 1)
        innovations = np.exp(theta[:n])
        dependence = theta[n:]
        factor = np.zeros((n, n))
        for row in range(n):
            factor[row, row] = innovations[row]
            multiplier = 1.0
            for column in range(row - 1, -1, -1):
                multiplier *= dependence[column]
                factor[row, column] = multiplier * innovations[column]
        return factor @ factor.T


class SpatialPower(CovarianceStructure):
    """Spatial-power covariance ``sd² * rho**distance`` for arbitrary spacing."""

    name = "spatial-power"

    def parameter_count(self, size: int | None = None) -> int:
        if self.dimension is not None or size is not None:
            _dimension(self.dimension, size, None)
        return 2

    def parameter_names(self, size: int | None = None) -> tuple[str, ...]:
        return ("log_sd", "logit_rho")

    def covariance(
        self,
        parameters: ArrayLike,
        *,
        size: int | None = None,
        index: ArrayLike | None = None,
        coordinates: ArrayLike | None = None,
        **_: Any,
    ) -> FloatArray:
        theta = _parameters(parameters, 2)
        points = coordinates if coordinates is not None else index
        if points is None:
            n = _dimension(self.dimension, size, None)
            locations = np.arange(n, dtype=float)[:, None]
        else:
            locations = np.asarray(points, dtype=float)
            if locations.ndim == 1:
                locations = locations[:, None]
            n = _dimension(self.dimension, size, locations)
            if locations.shape[0] != n or not np.all(np.isfinite(locations)):
                raise CovarianceError("Spatial coordinates are invalid.")
        distances = np.linalg.norm(locations[:, None, :] - locations[None, :, :], axis=2)
        rho = float(expit(theta[1]))
        correlation = _stabilize_correlation(rho**distances)
        return np.exp(2.0 * theta[0]) * correlation


class KnownCovariance(CovarianceStructure):
    """Fixed user-supplied covariance with no estimable parameters."""

    name = "known"

    def __init__(self, matrix: ArrayLike, *, group: str | None = None) -> None:
        value = np.asarray(matrix, dtype=float)
        validate_covariance(value, positive_semidefinite=True)
        super().__init__(value.shape[0], group=group)
        value.setflags(write=False)
        self.known_matrix = value

    def parameter_count(self, size: int | None = None) -> int:
        _dimension(self.dimension, size, None)
        return 0

    def covariance(
        self, parameters: ArrayLike = (), *, size: int | None = None, **_: Any
    ) -> FloatArray:
        _parameters(parameters, 0)
        _dimension(self.dimension, size, None)
        return self.known_matrix.copy()


DiagonalCovariance = Diagonal
UnstructuredCovariance = Unstructured
CompoundSymmetryCovariance = CompoundSymmetry
AR1Covariance = AR1
HeterogeneousAR1Covariance = HeterogeneousAR1
ToeplitzCovariance = Toeplitz
HeterogeneousToeplitzCovariance = HeterogeneousToeplitz
AnteDependenceCovariance = AnteDependence
SpatialPowerCovariance = SpatialPower


_BUILTINS: dict[str, type[CovarianceStructure]] = {
    "diagonal": Diagonal,
    "unstructured": Unstructured,
    "compound-symmetry": CompoundSymmetry,
    "cs": CompoundSymmetry,
    "ar1": AR1,
    "heterogeneous-ar1": HeterogeneousAR1,
    "heterogeneous_ar1": HeterogeneousAR1,
    "toeplitz": Toeplitz,
    "heterogeneous-toeplitz": HeterogeneousToeplitz,
    "heterogeneous_toeplitz": HeterogeneousToeplitz,
    "ante-dependence": AnteDependence,
    "ante_dependence": AnteDependence,
    "spatial-power": SpatialPower,
    "spatial_power": SpatialPower,
}

for _name, _implementation in _BUILTINS.items():
    if _name not in COVARIANCE_REGISTRY:
        COVARIANCE_REGISTRY.register(_name, _implementation, source="pymixef")


def covariance_structure(name: str, *args: Any, **kwargs: Any) -> CovarianceStructure:
    """Construct a built-in or registered covariance structure by name."""

    implementation = COVARIANCE_REGISTRY.get(name)
    value = implementation(*args, **kwargs)
    if not isinstance(value, CovarianceStructure):
        required = ("covariance", "validate", "simulate")
        if not all(hasattr(value, item) for item in required):
            raise CovarianceError(
                f"Registered covariance plugin {name!r} does not satisfy the protocol.",
                code="COV-PLUGIN-CONTRACT-001",
            )
    return value


get_covariance = covariance_structure


def singularity_report(matrix: ArrayLike, *, tolerance: float = 1e-8) -> dict[str, Any]:
    """Return non-raising rank, boundary, and near-perfect-correlation diagnostics."""

    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise CovarianceError("Covariance matrix must be square.", code="COV-SHAPE-001")
    eigenvalues = np.linalg.eigvalsh((value + value.T) / 2)
    maximum = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    effective_rank = int(np.sum(eigenvalues > tolerance * maximum))
    scales = np.sqrt(np.clip(np.diag(value), 0.0, None))
    denominator = scales[:, None] * scales[None, :]
    correlation = np.divide(value, denominator, out=np.zeros_like(value), where=denominator > 0)
    near_perfect: list[tuple[int, int, float]] = []
    for row in range(value.shape[0]):
        for column in range(row):
            if abs(correlation[row, column]) >= 1.0 - tolerance:
                near_perfect.append((column, row, float(correlation[row, column])))
    return {
        "eigenvalues": eigenvalues.tolist(),
        "effective_rank": effective_rank,
        "eigenvalue_ratio": float(eigenvalues[0] / maximum),
        "boundary_components": np.flatnonzero(scales <= np.sqrt(tolerance * maximum)).tolist(),
        "near_perfect_correlations": near_perfect,
        "singular": effective_rank < value.shape[0],
    }


__all__ = [
    "AR1",
    "AR1Covariance",
    "AnteDependence",
    "AnteDependenceCovariance",
    "CompoundSymmetry",
    "CompoundSymmetryCovariance",
    "CovarianceStructure",
    "CovarianceValidation",
    "Diagonal",
    "DiagonalCovariance",
    "HeterogeneousAR1",
    "HeterogeneousAR1Covariance",
    "HeterogeneousToeplitz",
    "HeterogeneousToeplitzCovariance",
    "KnownCovariance",
    "SpatialPower",
    "SpatialPowerCovariance",
    "Toeplitz",
    "ToeplitzCovariance",
    "Unstructured",
    "UnstructuredCovariance",
    "covariance_structure",
    "get_covariance",
    "singularity_report",
    "validate_covariance",
]
