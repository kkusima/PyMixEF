"""Built-in SciPy/NumPy reference estimation backends."""

from __future__ import annotations

from typing import Any

from .base import (
    Backend,
    BackendError,
    BackendInputError,
    BackendNumericalError,
    BackendUnsupportedError,
    validate_payload,
)
from .glmm import GLMMBackend, LaplaceGLMMBackend, fit_glmm
from .lmm import DenseLMMBackend, GaussianLMMBackend, LMMBackend, fit_lmm
from .mmrm import MMRMBackend, fit_mmrm

BUILTIN_BACKENDS: dict[str, type[Any]] = {
    "lmm": GaussianLMMBackend,
    "gaussian-lmm": GaussianLMMBackend,
    "dense-lmm": GaussianLMMBackend,
    "laplace": LaplaceGLMMBackend,
    "glmm": LaplaceGLMMBackend,
    "mmrm": MMRMBackend,
}


def get_backend(name: str) -> Backend:
    """Instantiate a built-in backend by stable engine name."""

    normalized = name.lower().replace("_", "-").replace(" ", "-")
    try:
        backend_type = BUILTIN_BACKENDS[normalized]
    except KeyError as exc:
        raise BackendUnsupportedError(
            f"unknown built-in backend {name!r}",
            details={"available_backends": sorted(BUILTIN_BACKENDS)},
        ) from exc
    return backend_type()


__all__ = [
    "BUILTIN_BACKENDS",
    "Backend",
    "BackendError",
    "BackendInputError",
    "BackendNumericalError",
    "BackendUnsupportedError",
    "DenseLMMBackend",
    "GLMMBackend",
    "GaussianLMMBackend",
    "LMMBackend",
    "LaplaceGLMMBackend",
    "MMRMBackend",
    "fit_glmm",
    "fit_lmm",
    "fit_mmrm",
    "get_backend",
    "validate_payload",
]
