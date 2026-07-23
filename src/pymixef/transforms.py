"""Canonical constraints between optimizer and natural parameter scales."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, logsumexp

from .errors import TransformError

FloatArray = NDArray[np.float64]


def _array(value: ArrayLike) -> FloatArray:
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)):
        raise TransformError("Transform inputs must be finite.")
    return result


def _softplus(value: FloatArray) -> FloatArray:
    return np.logaddexp(0.0, value)


def _softplus_inverse(value: FloatArray) -> FloatArray:
    if np.any(value <= 0.0):
        raise TransformError("Softplus inverse requires strictly positive values.")
    return value + np.log(-np.expm1(-value))


class Transform(ABC):
    """Abstract optimizer-to-natural-scale transform."""

    name: ClassVar[str]

    @abstractmethod
    def forward(self, unconstrained: ArrayLike) -> FloatArray:
        """Map an unconstrained value to its natural scale."""

    @abstractmethod
    def inverse(self, natural: ArrayLike) -> FloatArray:
        """Map a natural-scale value to its unconstrained scale."""

    @abstractmethod
    def log_abs_det_jacobian(self, unconstrained: ArrayLike) -> float:
        """Return the log absolute determinant of the forward Jacobian."""

    def jacobian(self, unconstrained: ArrayLike) -> FloatArray:
        """Return a finite-difference Jacobian for general vector transforms."""

        x = _array(unconstrained).reshape(-1)
        y = np.asarray(self.forward(x), dtype=float).reshape(-1)
        out = np.empty((y.size, x.size), dtype=float)
        step = np.cbrt(np.finfo(float).eps)
        for column in range(x.size):
            delta = step * max(1.0, abs(x[column]))
            upper = x.copy()
            lower = x.copy()
            upper[column] += delta
            lower[column] -= delta
            out[:, column] = (
                np.asarray(self.forward(upper)).reshape(-1)
                - np.asarray(self.forward(lower)).reshape(-1)
            ) / (2.0 * delta)
        return out


@dataclass(frozen=True, slots=True)
class IdentityTransform(Transform):
    """No constraint."""

    name: ClassVar[str] = "identity"

    def forward(self, unconstrained: ArrayLike) -> FloatArray:
        return _array(unconstrained).copy()

    def inverse(self, natural: ArrayLike) -> FloatArray:
        return _array(natural).copy()

    def log_abs_det_jacobian(self, unconstrained: ArrayLike) -> float:
        _array(unconstrained)
        return 0.0


@dataclass(frozen=True, slots=True)
class LogTransform(Transform):
    """Strictly-positive transform using ``exp``."""

    name: ClassVar[str] = "log"

    def forward(self, unconstrained: ArrayLike) -> FloatArray:
        x = _array(unconstrained)
        with np.errstate(over="raise"):
            try:
                return np.exp(x)
            except FloatingPointError as exc:
                raise TransformError("Exponential transform overflowed.") from exc

    def inverse(self, natural: ArrayLike) -> FloatArray:
        y = _array(natural)
        if np.any(y <= 0.0):
            raise TransformError("Log transform inverse requires positive values.")
        return np.log(y)

    def log_abs_det_jacobian(self, unconstrained: ArrayLike) -> float:
        return float(np.sum(_array(unconstrained)))


@dataclass(frozen=True, slots=True)
class SoftplusTransform(Transform):
    """Smooth strictly-positive transform with a near-linear upper tail."""

    name: ClassVar[str] = "softplus"

    def forward(self, unconstrained: ArrayLike) -> FloatArray:
        return _softplus(_array(unconstrained))

    def inverse(self, natural: ArrayLike) -> FloatArray:
        return _softplus_inverse(_array(natural))

    def log_abs_det_jacobian(self, unconstrained: ArrayLike) -> float:
        x = _array(unconstrained)
        return float(np.sum(-np.logaddexp(0.0, -x)))


@dataclass(frozen=True, slots=True)
class BoundedTransform(Transform):
    """Map to the open interval ``(lower, upper)`` using a logistic map."""

    lower: float = 0.0
    upper: float = 1.0
    name: ClassVar[str] = "bounded"

    def __post_init__(self) -> None:
        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            raise TransformError("Bounded transform limits must be finite.")
        if self.lower >= self.upper:
            raise TransformError("Bounded transform requires lower < upper.")

    def forward(self, unconstrained: ArrayLike) -> FloatArray:
        return self.lower + (self.upper - self.lower) * expit(_array(unconstrained))

    def inverse(self, natural: ArrayLike) -> FloatArray:
        y = _array(natural)
        if np.any((y <= self.lower) | (y >= self.upper)):
            raise TransformError("Bounded transform inverse requires interior values.")
        p = (y - self.lower) / (self.upper - self.lower)
        return np.log(p) - np.log1p(-p)

    def log_abs_det_jacobian(self, unconstrained: ArrayLike) -> float:
        x = _array(unconstrained)
        return float(
            x.size * np.log(self.upper - self.lower)
            + np.sum(-np.logaddexp(0.0, -x) - np.logaddexp(0.0, x))
        )


@dataclass(frozen=True, slots=True)
class SimplexTransform(Transform):
    """Additive-log-ratio map from ``R^(K-1)`` to a ``K``-simplex."""

    name: ClassVar[str] = "simplex"

    def forward(self, unconstrained: ArrayLike) -> FloatArray:
        x = _array(unconstrained).reshape(-1)
        extended = np.concatenate((x, np.zeros(1)))
        return np.exp(extended - logsumexp(extended))

    def inverse(self, natural: ArrayLike) -> FloatArray:
        y = _array(natural).reshape(-1)
        if y.size < 2 or np.any(y <= 0.0) or not np.isclose(np.sum(y), 1.0):
            raise TransformError("Simplex values must be positive and sum to one.")
        return np.log(y[:-1]) - np.log(y[-1])

    def log_abs_det_jacobian(self, unconstrained: ArrayLike) -> float:
        return float(np.sum(np.log(self.forward(unconstrained))))


@dataclass(frozen=True, slots=True)
class OrderedTransform(Transform):
    """Map a vector to a strictly increasing vector."""

    name: ClassVar[str] = "ordered"

    def forward(self, unconstrained: ArrayLike) -> FloatArray:
        x = _array(unconstrained).reshape(-1)
        if x.size == 0:
            return x
        return np.concatenate((x[:1], x[0] + np.cumsum(_softplus(x[1:]))))

    def inverse(self, natural: ArrayLike) -> FloatArray:
        y = _array(natural).reshape(-1)
        if y.size == 0:
            return y
        increments = np.diff(y)
        if np.any(increments <= 0.0):
            raise TransformError("Ordered transform inverse requires increasing values.")
        return np.concatenate((y[:1], _softplus_inverse(increments)))

    def log_abs_det_jacobian(self, unconstrained: ArrayLike) -> float:
        x = _array(unconstrained).reshape(-1)
        return float(np.sum(-np.logaddexp(0.0, -x[1:])))


@dataclass(frozen=True, slots=True)
class CholeskyCovarianceTransform(Transform):
    """Map packed unconstrained Cholesky entries to an SPD covariance matrix."""

    dimension: int
    name: ClassVar[str] = "cholesky-covariance"

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise TransformError("Covariance dimension must be positive.")

    @property
    def unconstrained_size(self) -> int:
        return self.dimension * (self.dimension + 1) // 2

    def factor(self, unconstrained: ArrayLike) -> FloatArray:
        x = _array(unconstrained).reshape(-1)
        if x.size != self.unconstrained_size:
            raise TransformError(f"Expected {self.unconstrained_size} packed values; got {x.size}.")
        lower = np.zeros((self.dimension, self.dimension), dtype=float)
        cursor = 0
        for row in range(self.dimension):
            for column in range(row + 1):
                value = x[cursor]
                lower[row, column] = np.exp(value) if row == column else value
                cursor += 1
        return lower

    def forward(self, unconstrained: ArrayLike) -> FloatArray:
        factor = self.factor(unconstrained)
        return factor @ factor.T

    def inverse(self, natural: ArrayLike) -> FloatArray:
        matrix = _array(natural)
        if matrix.shape != (self.dimension, self.dimension):
            raise TransformError("Covariance matrix has the wrong dimensions.")
        try:
            lower = np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as exc:
            raise TransformError("Covariance matrix must be positive definite.") from exc
        packed: list[float] = []
        for row in range(self.dimension):
            for column in range(row + 1):
                value = lower[row, column]
                packed.append(float(np.log(value) if row == column else value))
        return np.asarray(packed)

    def log_abs_det_jacobian(self, unconstrained: ArrayLike) -> float:
        lower = self.factor(unconstrained)
        powers = np.arange(self.dimension + 1, 1, -1, dtype=float)
        return float(self.dimension * np.log(2.0) + np.dot(powers, np.log(np.diag(lower))))


_TRANSFORMS: dict[str, type[Transform]] = {
    "identity": IdentityTransform,
    "log": LogTransform,
    "positive": LogTransform,
    "softplus": SoftplusTransform,
    "bounded": BoundedTransform,
    "simplex": SimplexTransform,
    "ordered": OrderedTransform,
    "cholesky-covariance": CholeskyCovarianceTransform,
}


def get_transform(name: str, **options: object) -> Transform:
    """Construct a canonical transform by stable name."""

    key = name.strip().lower().replace("_", "-")
    try:
        transform_type = _TRANSFORMS[key]
    except KeyError as exc:
        raise TransformError(
            f"Unknown transform {name!r}.",
            code="TRANSFORM-UNKNOWN-001",
            details={"available": sorted(_TRANSFORMS)},
        ) from exc
    return transform_type(**options)  # type: ignore[arg-type]


__all__ = [
    "BoundedTransform",
    "CholeskyCovarianceTransform",
    "IdentityTransform",
    "LogTransform",
    "OrderedTransform",
    "SimplexTransform",
    "SoftplusTransform",
    "Transform",
    "get_transform",
]
