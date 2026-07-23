"""Population-NLME estimation building blocks.

This module intentionally separates implemented numerical primitives from
production estimator claims.  It provides an interaction-aware conditional
mode objective, conditional-mode optimization, and a Laplace contribution
that can be consumed by a future FOCEI engine.  ``fit_focei`` is not silently
approximated and raises a stable unsupported-engine error.

A callback-based SAEM kernel is available with an explicit experimental label.
It implements random-walk Metropolis simulation and Robbins--Monro stochastic
approximation, while requiring callers to provide the model-specific
sufficient statistics and exact M-step.  This is a transparent algorithmic
kernel, not a claim of general NONMEM/Monolix equivalence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite, log, pi
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import log_ndtr

from .pk import ObservationError


class EstimationError(RuntimeError):
    """Base class for pharmacometric estimation failures."""

    code = "ESTIMATION-FAILED-001"


class UnsupportedEstimatorError(EstimationError):
    """Stable error for declared but not production-ready estimators."""

    code = "ENGINE-UNSUPPORTED-001"

    def __init__(self, engine: str, *, compatible: Sequence[str] = ()) -> None:
        self.engine = engine
        self.compatible = tuple(compatible)
        alternatives = (
            "" if not self.compatible else f"; available components: {', '.join(self.compatible)}"
        )
        super().__init__(
            f"{engine!r} is not implemented as a production fitting engine{alternatives}"
        )


class ConditionalModeError(EstimationError):
    """Raised for invalid or failed subject-level conditional objectives."""

    code = "NLME-CONDITIONAL-MODE-001"


class SAEMError(EstimationError):
    """Raised when the experimental SAEM kernel encounters an invalid callback."""

    code = "SAEM-EXPERIMENTAL-FAILED-001"


def _readonly(value: ArrayLike, *, ndim: int | None = None) -> NDArray[np.float64]:
    array = np.array(value, dtype=float, copy=True)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"expected {ndim} dimensions; got shape {array.shape}")
    array.setflags(write=False)
    return array


def _positive_definite(
    covariance: ArrayLike, *, name: str
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if np.any(~np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    try:
        cholesky = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    log_determinant = 2.0 * float(np.log(np.diag(cholesky)).sum())
    return matrix, cholesky, log_determinant


def apply_random_effects(
    typical_values: Mapping[str, float],
    eta: Mapping[str, float] | ArrayLike,
    *,
    names: Sequence[str] | None = None,
    relationships: Mapping[str, str] | None = None,
) -> Mapping[str, float]:
    """Apply common random-effect relationships to typical values.

    The default relationship is exponential, ``individual = typical*exp(eta)``.
    Per-parameter alternatives are ``additive`` and ``logit``.  The latter
    expects a typical value strictly between zero and one and adds ``eta`` on
    the logit scale.
    """

    typical = {str(name): float(value) for name, value in typical_values.items()}
    if isinstance(eta, Mapping):
        effects = {str(name): float(value) for name, value in eta.items()}
    else:
        if names is None:
            names = tuple(typical)
        eta_array = np.asarray(eta, dtype=float)
        if eta_array.shape != (len(names),):
            raise ValueError("eta length must match names")
        effects = dict(zip((str(name) for name in names), eta_array, strict=True))
    relation_map = {} if relationships is None else dict(relationships)
    individual: dict[str, float] = {}
    for name, value in typical.items():
        effect = effects.get(name, 0.0)
        relationship = relation_map.get(name, "exponential")
        if relationship == "exponential":
            result = value * np.exp(effect)
        elif relationship == "additive":
            result = value + effect
        elif relationship == "logit":
            if not 0 < value < 1:
                raise ValueError(f"logit relationship for {name!r} requires 0 < typical < 1")
            logit = np.log(value / (1.0 - value))
            result = 1.0 / (1.0 + np.exp(-(logit + effect)))
        else:
            raise ValueError(f"unsupported relationship {relationship!r} for parameter {name!r}")
        if not isfinite(float(result)):
            raise ValueError(f"random-effect transformation for {name!r} is non-finite")
        individual[name] = float(result)
    return MappingProxyType(individual)


def omega_from_standard_deviations(
    standard_deviations: ArrayLike, correlation: ArrayLike | None = None
) -> NDArray[np.float64]:
    """Construct a positive-definite random-effects covariance matrix."""

    sd = np.asarray(standard_deviations, dtype=float)
    if sd.ndim != 1 or sd.size == 0 or np.any(~np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError("standard_deviations must be a positive finite vector")
    if correlation is None:
        matrix = np.diag(sd * sd)
    else:
        corr = np.asarray(correlation, dtype=float)
        if corr.shape != (sd.size, sd.size):
            raise ValueError("correlation has incompatible shape")
        if not np.allclose(np.diag(corr), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("correlation must have a unit diagonal")
        matrix = sd[:, None] * corr * sd[None, :]
    _positive_definite(matrix, name="omega")
    return _readonly(matrix, ndim=2)


def eta_shrinkage(eta_estimates: ArrayLike, omega: ArrayLike) -> NDArray[np.float64]:
    """Variance-based ETA shrinkage ``1 - Var(eta_hat)/diag(Omega)``.

    The result is diagnostic and is not clipped; negative values reveal
    empirical ETA variance greater than the modeled population variance.
    """

    eta = np.asarray(eta_estimates, dtype=float)
    covariance, _, _ = _positive_definite(omega, name="omega")
    if eta.ndim != 2 or eta.shape[1] != covariance.shape[0]:
        raise ValueError("eta_estimates must have shape (subjects, eta_dimension)")
    if eta.shape[0] < 2:
        raise ValueError("at least two subjects are required for shrinkage")
    shrinkage = 1.0 - np.var(eta, axis=0, ddof=1) / np.diag(covariance)
    return _readonly(shrinkage, ndim=1)


@dataclass(frozen=True, slots=True)
class ObjectiveComponents:
    """Conditional objective decomposition at one ETA value."""

    total: float
    observation: float
    random_effect: float
    predictions: NDArray[np.float64]
    variances: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "predictions", _readonly(self.predictions, ndim=1))
        object.__setattr__(self, "variances", _readonly(self.variances, ndim=1))


@dataclass(frozen=True, slots=True)
class ConditionalObjective:
    """FOCEI-ready subject conditional negative log joint density.

    ``predict`` receives an ETA vector.  Residual variance is recomputed from
    each resulting prediction, retaining ETA--residual-variance interaction.
    Missing observations (NaN) are excluded explicitly.  A ``censored`` mask
    uses the Gaussian log-CDF at ``lower_limits`` (an M3-like contribution).
    """

    observations: NDArray[np.float64]
    predict: Callable[[NDArray[np.float64]], ArrayLike]
    omega: NDArray[np.float64]
    error: ObservationError
    error_parameters: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    censored: NDArray[np.bool_] | None = None
    lower_limits: NDArray[np.float64] | None = None
    include_constants: bool = True
    _omega_cholesky: NDArray[np.float64] = field(init=False, repr=False)
    _omega_logdet: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        observations = np.array(self.observations, dtype=float, copy=True)
        if observations.ndim != 1:
            raise ValueError("observations must be one-dimensional")
        omega, cholesky, logdet = _positive_definite(self.omega, name="omega")
        observations.setflags(write=False)
        omega = _readonly(omega, ndim=2)
        cholesky = _readonly(cholesky, ndim=2)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "_omega_cholesky", cholesky)
        object.__setattr__(self, "_omega_logdet", logdet)
        object.__setattr__(self, "error_parameters", MappingProxyType(dict(self.error_parameters)))
        if self.censored is not None:
            censored = np.array(self.censored, dtype=bool, copy=True)
            if censored.shape != observations.shape:
                raise ValueError("censored mask must match observations")
            censored.setflags(write=False)
            object.__setattr__(self, "censored", censored)
            if self.lower_limits is None:
                raise ValueError("censored observations require lower_limits")
        if self.lower_limits is not None:
            limits = np.array(self.lower_limits, dtype=float, copy=True)
            if limits.shape != observations.shape:
                raise ValueError("lower_limits must match observations")
            limits.setflags(write=False)
            object.__setattr__(self, "lower_limits", limits)

    @property
    def eta_dimension(self) -> int:
        return self.omega.shape[0]

    def components(self, eta: ArrayLike) -> ObjectiveComponents:
        eta_array = np.asarray(eta, dtype=float)
        if eta_array.shape != (self.eta_dimension,):
            raise ConditionalModeError(
                f"eta must have shape ({self.eta_dimension},), got {eta_array.shape}"
            )
        if np.any(~np.isfinite(eta_array)):
            raise ConditionalModeError("eta must contain finite values")
        predictions = np.asarray(self.predict(eta_array), dtype=float)
        if predictions.shape != self.observations.shape:
            raise ConditionalModeError(
                f"predict returned shape {predictions.shape}; expected {self.observations.shape}"
            )
        if np.any(~np.isfinite(predictions)):
            raise ConditionalModeError("predict returned non-finite values")
        variances = np.asarray(self.error.variance(predictions, self.error_parameters), dtype=float)
        if variances.ndim == 0:
            variances = np.full(predictions.shape, float(variances))
        if variances.shape != predictions.shape:
            try:
                variances = np.broadcast_to(variances, predictions.shape).copy()
            except ValueError as exc:
                raise ConditionalModeError(
                    "residual-error variance does not match predictions"
                ) from exc
        if np.any(~np.isfinite(variances)) or np.any(variances <= 0):
            raise ConditionalModeError("residual-error variances must be positive")

        observed_mask = np.isfinite(self.observations)
        censored = (
            np.zeros(self.observations.shape, dtype=bool)
            if self.censored is None
            else self.censored
        )
        uncensored_mask = observed_mask & ~censored
        residual = self.observations[uncensored_mask] - predictions[uncensored_mask]
        uncensored_variance = variances[uncensored_mask]
        observation_term = 0.5 * float(np.sum(np.square(residual) / uncensored_variance))
        observation_term += 0.5 * float(np.log(uncensored_variance).sum())
        if self.include_constants:
            observation_term += 0.5 * int(uncensored_mask.sum()) * log(2.0 * pi)
        if np.any(censored):
            assert self.lower_limits is not None
            standardized = (self.lower_limits[censored] - predictions[censored]) / np.sqrt(
                variances[censored]
            )
            observation_term -= float(log_ndtr(standardized).sum())

        whitened = np.linalg.solve(self._omega_cholesky, eta_array)
        random_effect_term = 0.5 * float(whitened @ whitened)
        random_effect_term += 0.5 * self._omega_logdet
        if self.include_constants:
            random_effect_term += 0.5 * self.eta_dimension * log(2.0 * pi)
        total = observation_term + random_effect_term
        if not isfinite(total):
            raise ConditionalModeError("conditional objective is non-finite")
        return ObjectiveComponents(
            total,
            observation_term,
            random_effect_term,
            predictions,
            variances,
        )

    def __call__(self, eta: ArrayLike) -> float:
        return self.components(eta).total


def conditional_mode_objective(
    eta: ArrayLike,
    *,
    observations: ArrayLike,
    predict: Callable[[NDArray[np.float64]], ArrayLike],
    omega: ArrayLike,
    error: ObservationError,
    error_parameters: Mapping[str, float] | None = None,
    censored: ArrayLike | None = None,
    lower_limits: ArrayLike | None = None,
    include_constants: bool = True,
) -> float:
    """Functional interface to :class:`ConditionalObjective`."""

    objective = ConditionalObjective(
        np.asarray(observations, dtype=float),
        predict,
        np.asarray(omega, dtype=float),
        error,
        {} if error_parameters is None else error_parameters,
        None if censored is None else np.asarray(censored, dtype=bool),
        None if lower_limits is None else np.asarray(lower_limits, dtype=float),
        include_constants,
    )
    return objective(eta)


def finite_difference_gradient(
    function: Callable[[NDArray[np.float64]], float],
    point: ArrayLike,
    *,
    relative_step: float = np.cbrt(np.finfo(float).eps),
) -> NDArray[np.float64]:
    """Central finite-difference gradient for derivative verification."""

    x = np.asarray(point, dtype=float)
    if x.ndim != 1:
        raise ValueError("point must be one-dimensional")
    gradient = np.empty_like(x)
    for index in range(x.size):
        increment = relative_step * max(1.0, abs(x[index]))
        plus = x.copy()
        minus = x.copy()
        plus[index] += increment
        minus[index] -= increment
        gradient[index] = (function(plus) - function(minus)) / (2.0 * increment)
    return _readonly(gradient, ndim=1)


def finite_difference_hessian(
    function: Callable[[NDArray[np.float64]], float],
    point: ArrayLike,
    *,
    relative_step: float = np.finfo(float).eps ** 0.25,
) -> NDArray[np.float64]:
    """Symmetric finite-difference Hessian for small conditional-mode problems."""

    x = np.asarray(point, dtype=float)
    if x.ndim != 1:
        raise ValueError("point must be one-dimensional")
    dimension = x.size
    hessian = np.empty((dimension, dimension), dtype=float)
    center = float(function(x))
    increments = relative_step * np.maximum(1.0, np.abs(x))
    for i in range(dimension):
        plus = x.copy()
        minus = x.copy()
        plus[i] += increments[i]
        minus[i] -= increments[i]
        hessian[i, i] = (function(plus) - 2.0 * center + function(minus)) / increments[i] ** 2
        for j in range(i):
            pp = x.copy()
            pm = x.copy()
            mp = x.copy()
            mm = x.copy()
            pp[i] += increments[i]
            pp[j] += increments[j]
            pm[i] += increments[i]
            pm[j] -= increments[j]
            mp[i] -= increments[i]
            mp[j] += increments[j]
            mm[i] -= increments[i]
            mm[j] -= increments[j]
            value = (function(pp) - function(pm) - function(mp) + function(mm)) / (
                4.0 * increments[i] * increments[j]
            )
            hessian[i, j] = value
            hessian[j, i] = value
    return _readonly(0.5 * (hessian + hessian.T), ndim=2)


@dataclass(frozen=True, slots=True)
class ConditionalModeResult:
    """Transparent subject-level optimizer result."""

    eta: NDArray[np.float64]
    objective: float
    observation_objective: float
    random_effect_objective: float
    gradient: NDArray[np.float64]
    hessian: NDArray[np.float64]
    covariance: NDArray[np.float64] | None
    success: bool
    message: str
    iterations: int
    function_evaluations: int
    gradient_norm: float
    hessian_positive_definite: bool
    warning_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "eta", _readonly(self.eta, ndim=1))
        object.__setattr__(self, "gradient", _readonly(self.gradient, ndim=1))
        object.__setattr__(self, "hessian", _readonly(self.hessian, ndim=2))
        if self.covariance is not None:
            object.__setattr__(self, "covariance", _readonly(self.covariance, ndim=2))


def find_conditional_mode(
    objective: ConditionalObjective,
    initial_eta: ArrayLike | None = None,
    *,
    method: str = "BFGS",
    tolerance: float = 1e-8,
    max_iterations: int = 500,
    require_success: bool = False,
) -> ConditionalModeResult:
    """Optimize a subject ETA mode and independently inspect its Hessian."""

    eta0 = (
        np.zeros(objective.eta_dimension, dtype=float)
        if initial_eta is None
        else np.asarray(initial_eta, dtype=float)
    )
    if eta0.shape != (objective.eta_dimension,):
        raise ValueError(f"initial_eta must have shape ({objective.eta_dimension},)")
    result = minimize(
        objective,
        eta0,
        method=method,
        tol=tolerance,
        options={"maxiter": int(max_iterations)},
    )
    eta = np.asarray(result.x, dtype=float)
    components = objective.components(eta)
    gradient = finite_difference_gradient(objective, eta)
    hessian = finite_difference_hessian(objective, eta)
    eigenvalues = np.linalg.eigvalsh(hessian)
    positive_definite = bool(np.all(eigenvalues > 0))
    covariance: NDArray[np.float64] | None
    warnings: list[str] = []
    if positive_definite:
        covariance = np.linalg.inv(hessian)
    else:
        covariance = None
        warnings.append("NLME-HESSIAN-NONPOSITIVE-001")
    gradient_norm = float(np.linalg.norm(gradient, ord=np.inf))
    if not result.success:
        warnings.append("NLME-CONDITIONAL-OPTIMIZER-001")
    if gradient_norm > max(1e-4, tolerance * 100.0):
        warnings.append("NLME-CONDITIONAL-GRADIENT-001")
    mode_result = ConditionalModeResult(
        eta,
        components.total,
        components.observation,
        components.random_effect,
        gradient,
        hessian,
        covariance,
        bool(result.success),
        str(result.message),
        int(getattr(result, "nit", 0)),
        int(getattr(result, "nfev", 0)),
        gradient_norm,
        positive_definite,
        tuple(warnings),
    )
    if require_success and (not mode_result.success or not positive_definite):
        raise ConditionalModeError(
            f"conditional mode failed: {mode_result.message}; warnings={mode_result.warning_codes}"
        )
    return mode_result


@dataclass(frozen=True, slots=True)
class LaplacePopulationResult:
    """Sum of subject Laplace contributions at conditional modes."""

    objective: float
    subject_contributions: NDArray[np.float64]
    modes: tuple[ConditionalModeResult, ...]
    warning_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "subject_contributions", _readonly(self.subject_contributions, ndim=1)
        )
        object.__setattr__(self, "modes", tuple(self.modes))


def laplace_population_objective(
    subject_objectives: Sequence[ConditionalObjective],
    *,
    initial_etas: Sequence[ArrayLike] | None = None,
    require_modes: bool = True,
    **mode_options: Any,
) -> LaplacePopulationResult:
    """Evaluate Laplace-integrated subject objectives.

    This is a FOCEI-ready outer-objective component.  It does not optimize or
    transform population parameters and is therefore not exposed as a complete
    FOCEI fit.
    """

    if initial_etas is not None and len(initial_etas) != len(subject_objectives):
        raise ValueError("initial_etas must match subject_objectives length")
    modes: list[ConditionalModeResult] = []
    contributions: list[float] = []
    warnings: list[str] = []
    for index, objective in enumerate(subject_objectives):
        initial = None if initial_etas is None else initial_etas[index]
        mode = find_conditional_mode(
            objective, initial, require_success=require_modes, **mode_options
        )
        modes.append(mode)
        sign, logdet = np.linalg.slogdet(mode.hessian)
        if sign <= 0:
            if require_modes:
                raise ConditionalModeError(
                    f"subject {index} conditional Hessian is not positive definite"
                )
            contribution = np.inf
        else:
            q = mode.eta.size
            contribution = mode.objective + 0.5 * logdet - 0.5 * q * log(2.0 * pi)
        contributions.append(float(contribution))
        warnings.extend(mode.warning_codes)
    contribution_array = np.asarray(contributions, dtype=float)
    return LaplacePopulationResult(
        float(contribution_array.sum()),
        contribution_array,
        tuple(modes),
        tuple(dict.fromkeys(warnings)),
    )


def fit_focei(*args: Any, **kwargs: Any) -> None:
    """Refuse a production FOCEI claim until the complete engine is validated."""

    del args, kwargs
    raise UnsupportedEstimatorError(
        "focei",
        compatible=(
            "ConditionalObjective",
            "find_conditional_mode",
            "laplace_population_objective",
        ),
    )


@dataclass(frozen=True, slots=True)
class SAEMProblem:
    """Model-specific callbacks required by the experimental SAEM kernel.

    ``log_joint(parameters, latent)`` must return the log of the target density
    up to a constant.  ``sufficient_statistics(parameters, latent)`` returns a
    fixed-shape numeric vector/array.  ``m_step(averaged_statistics,
    current_parameters)`` performs the exact model-specific maximization.
    """

    initial_parameters: NDArray[np.float64]
    initial_latent: NDArray[np.float64]
    log_joint: Callable[[NDArray[np.float64], NDArray[np.float64]], float]
    sufficient_statistics: Callable[[NDArray[np.float64], NDArray[np.float64]], ArrayLike]
    m_step: Callable[[NDArray[np.float64], NDArray[np.float64]], ArrayLike]
    parameter_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        parameters = np.array(self.initial_parameters, dtype=float, copy=True)
        latent = np.array(self.initial_latent, dtype=float, copy=True)
        if parameters.ndim != 1 or parameters.size == 0:
            raise ValueError("initial_parameters must be a non-empty vector")
        if latent.size == 0:
            raise ValueError("initial_latent must be non-empty")
        if np.any(~np.isfinite(parameters)) or np.any(~np.isfinite(latent)):
            raise ValueError("SAEM initial values must be finite")
        if self.parameter_names and len(self.parameter_names) != parameters.size:
            raise ValueError("parameter_names must match initial_parameters")
        parameters.setflags(write=False)
        latent.setflags(write=False)
        object.__setattr__(self, "initial_parameters", parameters)
        object.__setattr__(self, "initial_latent", latent)
        object.__setattr__(self, "parameter_names", tuple(self.parameter_names))


@dataclass(frozen=True, slots=True)
class SAEMControl:
    """Versioned controls for the experimental SAEM kernel."""

    iterations: int = 1000
    burn_in: int = 300
    step_exponent: float = 0.7
    mcmc_steps: int = 2
    proposal_scale: float = 0.2
    seed: int = 20260722
    keep_latent_trace: bool = False

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.burn_in < 0 or self.burn_in >= self.iterations:
            raise ValueError("burn_in must be in [0, iterations)")
        if not 0.5 < self.step_exponent <= 1.0:
            raise ValueError("step_exponent must be in (0.5, 1]")
        if self.mcmc_steps <= 0:
            raise ValueError("mcmc_steps must be positive")
        if not isfinite(self.proposal_scale) or self.proposal_scale <= 0:
            raise ValueError("proposal_scale must be finite and positive")


@dataclass(frozen=True, slots=True)
class SAEMResult:
    """Trace and diagnostics from the explicitly experimental SAEM kernel."""

    parameters: NDArray[np.float64]
    latent: NDArray[np.float64]
    sufficient_statistics: NDArray[np.float64]
    parameter_trace: NDArray[np.float64]
    latent_trace: NDArray[np.float64] | None
    step_sizes: NDArray[np.float64]
    acceptance_rate: float
    accepted: int
    proposals: int
    seed: int
    burn_in: int
    step_exponent: float
    experimental: bool = True
    reproducibility_class: str = "stochastic-with-monte-carlo-error"
    warning_codes: tuple[str, ...] = ("SAEM-EXPERIMENTAL-001",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _readonly(self.parameters, ndim=1))
        latent = _readonly(self.latent)
        object.__setattr__(self, "latent", latent)
        object.__setattr__(self, "sufficient_statistics", _readonly(self.sufficient_statistics))
        object.__setattr__(self, "parameter_trace", _readonly(self.parameter_trace, ndim=2))
        if self.latent_trace is not None:
            object.__setattr__(self, "latent_trace", _readonly(self.latent_trace))
        object.__setattr__(self, "step_sizes", _readonly(self.step_sizes, ndim=1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": self.parameters.tolist(),
            "latent": self.latent.tolist(),
            "sufficient_statistics": self.sufficient_statistics.tolist(),
            "parameter_trace": self.parameter_trace.tolist(),
            "step_sizes": self.step_sizes.tolist(),
            "acceptance_rate": self.acceptance_rate,
            "accepted": self.accepted,
            "proposals": self.proposals,
            "seed": self.seed,
            "burn_in": self.burn_in,
            "step_exponent": self.step_exponent,
            "experimental": self.experimental,
            "reproducibility_class": self.reproducibility_class,
            "warning_codes": list(self.warning_codes),
        }


def experimental_saem(problem: SAEMProblem, control: SAEMControl | None = None) -> SAEMResult:
    """Run the callback-based experimental SAEM algorithm.

    At iteration ``k`` the stochastic-approximation step is 1 during burn-in
    and ``(k - burn_in) ** (-step_exponent)`` afterward.  The latent simulation
    is a symmetric random-walk Metropolis kernel, making its acceptance ratio
    explicit and auditable.
    """

    controls = SAEMControl() if control is None else control
    rng = np.random.default_rng(controls.seed)
    parameters = np.array(problem.initial_parameters, copy=True)
    latent = np.array(problem.initial_latent, copy=True)
    parameter_trace = np.empty((controls.iterations, parameters.size), dtype=float)
    latent_trace = (
        np.empty((controls.iterations, *latent.shape), dtype=float)
        if controls.keep_latent_trace
        else None
    )
    step_sizes = np.empty(controls.iterations, dtype=float)
    accepted = 0
    proposals = 0

    try:
        current_log_joint = float(problem.log_joint(parameters, latent))
    except Exception as exc:
        raise SAEMError(f"log_joint failed at initialization: {exc}") from exc
    if not isfinite(current_log_joint):
        raise SAEMError("initial log_joint must be finite")

    averaged_statistics: NDArray[np.float64] | None = None
    statistic_shape: tuple[int, ...] | None = None
    for iteration in range(1, controls.iterations + 1):
        for _ in range(controls.mcmc_steps):
            proposal = latent + rng.normal(0.0, controls.proposal_scale, size=latent.shape)
            try:
                proposed_log_joint = float(problem.log_joint(parameters, proposal))
            except Exception as exc:
                raise SAEMError(f"log_joint failed at iteration {iteration}: {exc}") from exc
            proposals += 1
            if isfinite(proposed_log_joint) and (
                proposed_log_joint >= current_log_joint
                or log(rng.uniform()) < proposed_log_joint - current_log_joint
            ):
                latent = proposal
                current_log_joint = proposed_log_joint
                accepted += 1

        try:
            statistics = np.asarray(problem.sufficient_statistics(parameters, latent), dtype=float)
        except Exception as exc:
            raise SAEMError(
                f"sufficient_statistics failed at iteration {iteration}: {exc}"
            ) from exc
        if statistics.size == 0 or np.any(~np.isfinite(statistics)):
            raise SAEMError("sufficient statistics must be non-empty and finite")
        if statistic_shape is None:
            statistic_shape = statistics.shape
        elif statistics.shape != statistic_shape:
            raise SAEMError("sufficient-statistic shape changed between iterations")

        gamma = (
            1.0
            if iteration <= controls.burn_in
            else (iteration - controls.burn_in) ** (-controls.step_exponent)
        )
        step_sizes[iteration - 1] = gamma
        if averaged_statistics is None:
            averaged_statistics = np.array(statistics, copy=True)
        else:
            averaged_statistics += gamma * (statistics - averaged_statistics)
        try:
            updated = np.asarray(problem.m_step(averaged_statistics, parameters), dtype=float)
        except Exception as exc:
            raise SAEMError(f"m_step failed at iteration {iteration}: {exc}") from exc
        if updated.shape != parameters.shape or np.any(~np.isfinite(updated)):
            raise SAEMError("m_step must return a finite vector matching initial_parameters")
        parameters = updated
        # Parameters changed, so refresh the current latent target before the
        # next Metropolis ratio.
        current_log_joint = float(problem.log_joint(parameters, latent))
        if not isfinite(current_log_joint):
            raise SAEMError(f"log_joint became non-finite after M-step at iteration {iteration}")
        parameter_trace[iteration - 1] = parameters
        if latent_trace is not None:
            latent_trace[iteration - 1] = latent

    assert averaged_statistics is not None
    acceptance_rate = accepted / proposals
    warnings = ["SAEM-EXPERIMENTAL-001"]
    if acceptance_rate < 0.1 or acceptance_rate > 0.8:
        warnings.append("SAEM-MCMC-ACCEPTANCE-001")
    return SAEMResult(
        parameters,
        latent,
        averaged_statistics,
        parameter_trace,
        latent_trace,
        step_sizes,
        acceptance_rate,
        accepted,
        proposals,
        controls.seed,
        controls.burn_in,
        controls.step_exponent,
        warning_codes=tuple(warnings),
    )


saem = experimental_saem


__all__ = [
    "ConditionalModeError",
    "ConditionalModeResult",
    "ConditionalObjective",
    "EstimationError",
    "LaplacePopulationResult",
    "ObjectiveComponents",
    "SAEMControl",
    "SAEMError",
    "SAEMProblem",
    "SAEMResult",
    "UnsupportedEstimatorError",
    "apply_random_effects",
    "conditional_mode_objective",
    "eta_shrinkage",
    "experimental_saem",
    "find_conditional_mode",
    "finite_difference_gradient",
    "finite_difference_hessian",
    "fit_focei",
    "laplace_population_objective",
    "omega_from_standard_deviations",
    "saem",
]
