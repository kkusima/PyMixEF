"""Closed-form pharmacokinetic helpers and residual-error models.

The PK functions use amount, time, clearance, and volume on any mutually
consistent unit system.  Returned concentrations have units of amount/volume.
Inputs are NumPy-broadcastable and output ordering follows the input time
array.  Times before a dose (including lag time) return zero.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import log, pi
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import log_ndtr


class PKValidationError(ValueError):
    """Raised when PK parameters are outside their mathematical domain."""

    code = "PK-INVALID-001"


Numeric: TypeAlias = float | int | np.floating
ErrorParameter: TypeAlias = Numeric | str | Any


def _positive(name: str, value: float) -> float:
    converted = float(value)
    if not np.isfinite(converted) or converted <= 0:
        raise PKValidationError(f"{name} must be finite and strictly positive")
    return converted


def _nonnegative(name: str, value: float) -> float:
    converted = float(value)
    if not np.isfinite(converted) or converted < 0:
        raise PKValidationError(f"{name} must be finite and non-negative")
    return converted


def _times(time: ArrayLike, lag: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = np.asarray(time, dtype=float)
    if not np.all(np.isfinite(values)):
        raise PKValidationError("time must contain only finite values")
    relative = values - float(lag)
    return values, relative


def _return_like_time(time: ArrayLike, values: NDArray[np.float64]) -> float | NDArray[np.float64]:
    if np.asarray(time).ndim == 0:
        return float(np.asarray(values))
    return values


def one_compartment_iv_bolus(
    time: ArrayLike,
    *,
    dose: float,
    clearance: float,
    volume: float,
    bioavailability: float = 1.0,
    lag: float = 0.0,
) -> float | NDArray[np.float64]:
    """Concentration after a single IV bolus in a one-compartment model."""

    dose = _nonnegative("dose", dose)
    clearance = _positive("clearance", clearance)
    volume = _positive("volume", volume)
    bioavailability = _nonnegative("bioavailability", bioavailability)
    lag = _nonnegative("lag", lag)
    _, relative = _times(time, lag)
    k = clearance / volume
    concentration = np.where(
        relative >= 0,
        bioavailability * dose / volume * np.exp(-k * np.maximum(relative, 0.0)),
        0.0,
    )
    return _return_like_time(time, concentration)


def one_compartment_infusion(
    time: ArrayLike,
    *,
    clearance: float,
    volume: float,
    dose: float | None = None,
    rate: float | None = None,
    duration: float | None = None,
    bioavailability: float = 1.0,
    start: float = 0.0,
) -> float | NDArray[np.float64]:
    """Concentration for a finite constant-rate one-compartment infusion.

    Supply any two consistent values among ``dose``, ``rate``, and
    ``duration``.  If all three are supplied, ``dose == rate * duration`` is
    checked.
    """

    clearance = _positive("clearance", clearance)
    volume = _positive("volume", volume)
    bioavailability = _nonnegative("bioavailability", bioavailability)
    start = float(start)
    if not np.isfinite(start):
        raise PKValidationError("start must be finite")
    dose, rate, duration = _resolve_infusion(dose, rate, duration)
    _, relative = _times(time, start)
    k = clearance / volume
    active = np.clip(relative, 0.0, duration)
    concentration = (
        bioavailability
        * rate
        / clearance
        * (1.0 - np.exp(-k * active))
        * np.where(relative > duration, np.exp(-k * (relative - duration)), 1.0)
    )
    concentration = np.where(relative >= 0, concentration, 0.0)
    return _return_like_time(time, concentration)


def one_compartment_oral(
    time: ArrayLike,
    *,
    dose: float,
    clearance: float,
    volume: float,
    absorption_rate: float,
    bioavailability: float = 1.0,
    lag: float = 0.0,
) -> float | NDArray[np.float64]:
    """Concentration after a first-order oral dose.

    The numerically stable limiting expression is used when absorption and
    elimination rate constants are nearly equal.
    """

    dose = _nonnegative("dose", dose)
    clearance = _positive("clearance", clearance)
    volume = _positive("volume", volume)
    ka = _positive("absorption_rate", absorption_rate)
    bioavailability = _nonnegative("bioavailability", bioavailability)
    lag = _nonnegative("lag", lag)
    _, relative = _times(time, lag)
    t = np.maximum(relative, 0.0)
    ke = clearance / volume
    if np.isclose(ka, ke, rtol=1e-8, atol=1e-12):
        concentration = bioavailability * dose / volume * ka * t * np.exp(-ke * t)
    else:
        concentration = (
            bioavailability * dose / volume * ka / (ka - ke) * (np.exp(-ke * t) - np.exp(-ka * t))
        )
    concentration = np.where(relative >= 0, concentration, 0.0)
    return _return_like_time(time, concentration)


@dataclass(frozen=True, slots=True)
class TwoCompartmentRates:
    """Micro- and macro-rate constants for a linear two-compartment model."""

    k10: float
    k12: float
    k21: float
    alpha: float
    beta: float


def two_compartment_rates(
    *,
    clearance: float,
    central_volume: float,
    intercompartmental_clearance: float,
    peripheral_volume: float,
) -> TwoCompartmentRates:
    """Calculate microconstants and hybrid exponents ``alpha`` and ``beta``."""

    cl = _positive("clearance", clearance)
    v1 = _positive("central_volume", central_volume)
    q = _positive("intercompartmental_clearance", intercompartmental_clearance)
    v2 = _positive("peripheral_volume", peripheral_volume)
    k10 = cl / v1
    k12 = q / v1
    k21 = q / v2
    total = k10 + k12 + k21
    discriminant = max(total * total - 4.0 * k10 * k21, 0.0)
    root = np.sqrt(discriminant)
    alpha = 0.5 * (total + root)
    beta = 0.5 * (total - root)
    if alpha <= 0 or beta <= 0 or alpha < beta:
        raise PKValidationError("two-compartment hybrid rates are not positive and ordered")
    return TwoCompartmentRates(k10, k12, k21, float(alpha), float(beta))


def two_compartment_iv_bolus(
    time: ArrayLike,
    *,
    dose: float,
    clearance: float,
    central_volume: float,
    intercompartmental_clearance: float,
    peripheral_volume: float,
    bioavailability: float = 1.0,
    lag: float = 0.0,
) -> float | NDArray[np.float64]:
    """Central concentration after a single two-compartment IV bolus."""

    dose = _nonnegative("dose", dose)
    v1 = _positive("central_volume", central_volume)
    bioavailability = _nonnegative("bioavailability", bioavailability)
    lag = _nonnegative("lag", lag)
    rates = two_compartment_rates(
        clearance=clearance,
        central_volume=v1,
        intercompartmental_clearance=intercompartmental_clearance,
        peripheral_volume=peripheral_volume,
    )
    _, relative = _times(time, lag)
    t = np.maximum(relative, 0.0)
    denominator = rates.alpha - rates.beta
    coefficient_alpha = (rates.alpha - rates.k21) / denominator
    coefficient_beta = (rates.k21 - rates.beta) / denominator
    concentration = (
        bioavailability
        * dose
        / v1
        * (
            coefficient_alpha * np.exp(-rates.alpha * t)
            + coefficient_beta * np.exp(-rates.beta * t)
        )
    )
    concentration = np.where(relative >= 0, concentration, 0.0)
    return _return_like_time(time, concentration)


def two_compartment_infusion(
    time: ArrayLike,
    *,
    clearance: float,
    central_volume: float,
    intercompartmental_clearance: float,
    peripheral_volume: float,
    dose: float | None = None,
    rate: float | None = None,
    duration: float | None = None,
    bioavailability: float = 1.0,
    start: float = 0.0,
) -> float | NDArray[np.float64]:
    """Central concentration during and after a finite two-compartment infusion."""

    v1 = _positive("central_volume", central_volume)
    bioavailability = _nonnegative("bioavailability", bioavailability)
    dose, rate, duration = _resolve_infusion(dose, rate, duration)
    rates = two_compartment_rates(
        clearance=clearance,
        central_volume=v1,
        intercompartmental_clearance=intercompartmental_clearance,
        peripheral_volume=peripheral_volume,
    )
    _, relative = _times(time, start)
    elapsed = np.clip(relative, 0.0, duration)
    denominator = rates.alpha - rates.beta
    coefficient_alpha = (rates.alpha - rates.k21) / denominator
    coefficient_beta = (rates.k21 - rates.beta) / denominator

    # Integral of each exponential bolus-response term over the active input
    # interval.  After the infusion ends, both terms decay from their end value.
    alpha_component = (
        coefficient_alpha
        * (1.0 - np.exp(-rates.alpha * elapsed))
        / rates.alpha
        * np.where(relative > duration, np.exp(-rates.alpha * (relative - duration)), 1.0)
    )
    beta_component = (
        coefficient_beta
        * (1.0 - np.exp(-rates.beta * elapsed))
        / rates.beta
        * np.where(relative > duration, np.exp(-rates.beta * (relative - duration)), 1.0)
    )
    concentration = bioavailability * rate / v1 * (alpha_component + beta_component)
    concentration = np.where(relative >= 0, concentration, 0.0)
    return _return_like_time(time, concentration)


def two_compartment_oral(
    time: ArrayLike,
    *,
    dose: float,
    clearance: float,
    central_volume: float,
    intercompartmental_clearance: float,
    peripheral_volume: float,
    absorption_rate: float,
    bioavailability: float = 1.0,
    lag: float = 0.0,
) -> float | NDArray[np.float64]:
    """Central concentration after first-order absorption into two compartments."""

    dose = _nonnegative("dose", dose)
    v1 = _positive("central_volume", central_volume)
    ka = _positive("absorption_rate", absorption_rate)
    bioavailability = _nonnegative("bioavailability", bioavailability)
    lag = _nonnegative("lag", lag)
    rates = two_compartment_rates(
        clearance=clearance,
        central_volume=v1,
        intercompartmental_clearance=intercompartmental_clearance,
        peripheral_volume=peripheral_volume,
    )
    _, relative = _times(time, lag)
    t = np.maximum(relative, 0.0)
    denominator = rates.alpha - rates.beta
    coefficient_alpha = (rates.alpha - rates.k21) / denominator
    coefficient_beta = (rates.k21 - rates.beta) / denominator

    def absorbed_component(exponent: float) -> NDArray[np.float64]:
        if np.isclose(ka, exponent, rtol=1e-8, atol=1e-12):
            return ka * t * np.exp(-ka * t)
        return ka * (np.exp(-exponent * t) - np.exp(-ka * t)) / (ka - exponent)

    concentration = (
        bioavailability
        * dose
        / v1
        * (
            coefficient_alpha * absorbed_component(rates.alpha)
            + coefficient_beta * absorbed_component(rates.beta)
        )
    )
    concentration = np.where(relative >= 0, concentration, 0.0)
    return _return_like_time(time, concentration)


def _resolve_infusion(
    dose: float | None, rate: float | None, duration: float | None
) -> tuple[float, float, float]:
    supplied = sum(value is not None for value in (dose, rate, duration))
    if supplied < 2:
        raise PKValidationError("supply at least two of dose, rate, and duration")
    if dose is not None:
        dose = _nonnegative("dose", dose)
    if rate is not None:
        rate = _positive("rate", rate)
    if duration is not None:
        duration = _positive("duration", duration)
    if dose is None:
        assert rate is not None and duration is not None
        dose = rate * duration
    elif rate is None:
        assert duration is not None
        rate = dose / duration
    elif duration is None:
        duration = dose / rate
    elif not np.isclose(dose, rate * duration, rtol=1e-9, atol=1e-12):
        raise PKValidationError("dose, rate, and duration are inconsistent")
    return dose, rate, duration


@dataclass(frozen=True, slots=True)
class OneCompartmentPK:
    """Reusable parameter bundle for one-compartment closed forms."""

    clearance: float
    volume: float

    def __post_init__(self) -> None:
        _positive("clearance", self.clearance)
        _positive("volume", self.volume)

    def iv_bolus(self, time: ArrayLike, dose: float, **kwargs: float) -> Any:
        return one_compartment_iv_bolus(
            time,
            dose=dose,
            clearance=self.clearance,
            volume=self.volume,
            **kwargs,
        )

    def infusion(self, time: ArrayLike, **kwargs: float) -> Any:
        return one_compartment_infusion(
            time, clearance=self.clearance, volume=self.volume, **kwargs
        )

    def oral(self, time: ArrayLike, dose: float, absorption_rate: float, **kwargs: float) -> Any:
        return one_compartment_oral(
            time,
            dose=dose,
            clearance=self.clearance,
            volume=self.volume,
            absorption_rate=absorption_rate,
            **kwargs,
        )


@dataclass(frozen=True, slots=True)
class TwoCompartmentPK:
    """Reusable parameter bundle for two-compartment closed forms."""

    clearance: float
    central_volume: float
    intercompartmental_clearance: float
    peripheral_volume: float

    def __post_init__(self) -> None:
        two_compartment_rates(
            clearance=self.clearance,
            central_volume=self.central_volume,
            intercompartmental_clearance=self.intercompartmental_clearance,
            peripheral_volume=self.peripheral_volume,
        )

    def iv_bolus(self, time: ArrayLike, dose: float, **kwargs: float) -> Any:
        return two_compartment_iv_bolus(
            time,
            dose=dose,
            clearance=self.clearance,
            central_volume=self.central_volume,
            intercompartmental_clearance=self.intercompartmental_clearance,
            peripheral_volume=self.peripheral_volume,
            **kwargs,
        )

    def infusion(self, time: ArrayLike, **kwargs: float) -> Any:
        return two_compartment_infusion(
            time,
            clearance=self.clearance,
            central_volume=self.central_volume,
            intercompartmental_clearance=self.intercompartmental_clearance,
            peripheral_volume=self.peripheral_volume,
            **kwargs,
        )

    def oral(self, time: ArrayLike, dose: float, absorption_rate: float, **kwargs: float) -> Any:
        return two_compartment_oral(
            time,
            dose=dose,
            clearance=self.clearance,
            central_volume=self.central_volume,
            intercompartmental_clearance=self.intercompartmental_clearance,
            peripheral_volume=self.peripheral_volume,
            absorption_rate=absorption_rate,
            **kwargs,
        )


def _parameter_value(parameter: ErrorParameter, values: Mapping[str, float] | None) -> float:
    if isinstance(parameter, str):
        if values is None:
            raise PKValidationError(
                f"error parameter {parameter!r} is symbolic; supply a values mapping"
            )
        if parameter not in values:
            raise KeyError(parameter)
        return _nonnegative(parameter, values[parameter])
    # DSL Param objects are intentionally accepted without importing dsl.py.
    name = getattr(parameter, "name", None)
    if name is not None and not isinstance(parameter, (int, float, np.number)):
        if values is None:
            raise PKValidationError(
                f"error parameter {name!r} is symbolic; supply a values mapping"
            )
        return _nonnegative(str(name), values[str(name)])
    return _nonnegative("error parameter", float(parameter))


@runtime_checkable
class ObservationError(Protocol):
    """Protocol implemented by residual-error declarations."""

    def variance(
        self, prediction: ArrayLike, parameters: Mapping[str, float] | None = None
    ) -> NDArray[np.float64]: ...

    def to_dict(self) -> dict[str, Any]: ...


class _ErrorOperations:
    def __add__(self, other: ObservationError) -> CombinedError:
        if not isinstance(other, ObservationError):
            return NotImplemented
        return CombinedError.from_components(self, other)

    def logpdf(
        self,
        observed: ArrayLike,
        prediction: ArrayLike,
        parameters: Mapping[str, float] | None = None,
    ) -> NDArray[np.float64]:
        observed_array = np.asarray(observed, dtype=float)
        prediction_array = np.asarray(prediction, dtype=float)
        variance = self.variance(prediction_array, parameters)  # type: ignore[attr-defined]
        return -0.5 * (
            np.log(2.0 * pi * variance) + np.square(observed_array - prediction_array) / variance
        )

    def simulate(
        self,
        prediction: ArrayLike,
        *,
        rng: np.random.Generator | None = None,
        parameters: Mapping[str, float] | None = None,
    ) -> NDArray[np.float64]:
        generator = np.random.default_rng() if rng is None else rng
        prediction_array = np.asarray(prediction, dtype=float)
        standard_deviation = np.sqrt(self.variance(prediction_array, parameters))  # type: ignore[attr-defined]
        return prediction_array + generator.normal(size=prediction_array.shape) * standard_deviation


@dataclass(frozen=True, slots=True)
class AdditiveError(_ErrorOperations):
    """Gaussian additive residual standard deviation."""

    sigma: ErrorParameter
    variance_floor: float = np.finfo(float).tiny

    def variance(
        self, prediction: ArrayLike, parameters: Mapping[str, float] | None = None
    ) -> NDArray[np.float64]:
        sigma = _parameter_value(self.sigma, parameters)
        shape = np.asarray(prediction, dtype=float).shape
        return np.full(shape, max(sigma * sigma, self.variance_floor), dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "additive", "sigma": getattr(self.sigma, "name", self.sigma)}


@dataclass(frozen=True, slots=True)
class ProportionalError(_ErrorOperations):
    """Gaussian residual SD equal to ``sigma * abs(prediction)``."""

    sigma: ErrorParameter
    variance_floor: float = np.finfo(float).tiny

    def variance(
        self, prediction: ArrayLike, parameters: Mapping[str, float] | None = None
    ) -> NDArray[np.float64]:
        sigma = _parameter_value(self.sigma, parameters)
        prediction_array = np.asarray(prediction, dtype=float)
        return np.maximum(np.square(sigma * prediction_array), self.variance_floor)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "proportional", "sigma": getattr(self.sigma, "name", self.sigma)}


@dataclass(frozen=True, slots=True)
class PowerError(_ErrorOperations):
    """Gaussian residual SD ``sigma * abs(prediction) ** power``."""

    sigma: ErrorParameter
    power: ErrorParameter = 1.0
    variance_floor: float = np.finfo(float).tiny

    def variance(
        self, prediction: ArrayLike, parameters: Mapping[str, float] | None = None
    ) -> NDArray[np.float64]:
        sigma = _parameter_value(self.sigma, parameters)
        exponent = _parameter_value(self.power, parameters)
        prediction_array = np.asarray(prediction, dtype=float)
        return np.maximum(
            np.square(sigma * np.power(np.abs(prediction_array), exponent)),
            self.variance_floor,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "power",
            "sigma": getattr(self.sigma, "name", self.sigma),
            "power": getattr(self.power, "name", self.power),
        }


@dataclass(frozen=True, slots=True)
class CombinedError(_ErrorOperations):
    """Independent additive and prediction-dependent Gaussian errors.

    Component variances are added; standard deviations are not.
    """

    additive_sigma: ErrorParameter
    proportional_sigma: ErrorParameter
    power: ErrorParameter = 1.0
    variance_floor: float = np.finfo(float).tiny

    @classmethod
    def from_components(cls, first: ObservationError, second: ObservationError) -> CombinedError:
        components = (first, second)
        additive_component = next(
            (item for item in components if isinstance(item, AdditiveError)), None
        )
        power_component = next(
            (item for item in components if isinstance(item, (ProportionalError, PowerError))),
            None,
        )
        if additive_component is None or power_component is None:
            raise TypeError("only additive + proportional/power errors form a CombinedError")
        if isinstance(power_component, ProportionalError):
            return cls(additive_component.sigma, power_component.sigma, 1.0)
        return cls(additive_component.sigma, power_component.sigma, power_component.power)

    def variance(
        self, prediction: ArrayLike, parameters: Mapping[str, float] | None = None
    ) -> NDArray[np.float64]:
        additive_sigma = _parameter_value(self.additive_sigma, parameters)
        proportional_sigma = _parameter_value(self.proportional_sigma, parameters)
        exponent = _parameter_value(self.power, parameters)
        prediction_array = np.asarray(prediction, dtype=float)
        return np.maximum(
            additive_sigma**2
            + np.square(proportional_sigma * np.power(np.abs(prediction_array), exponent)),
            self.variance_floor,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "combined",
            "additive_sigma": getattr(self.additive_sigma, "name", self.additive_sigma),
            "proportional_sigma": getattr(self.proportional_sigma, "name", self.proportional_sigma),
            "power": getattr(self.power, "name", self.power),
        }


@dataclass(frozen=True, slots=True)
class LogNormalError:
    """Multiplicative log-normal observation error.

    ``log(observed) ~ Normal(log(prediction), sigma)``.  ``logpdf`` includes
    the Jacobian for density on the original observation scale.
    """

    sigma: ErrorParameter

    def variance(
        self, prediction: ArrayLike, parameters: Mapping[str, float] | None = None
    ) -> NDArray[np.float64]:
        sigma = _parameter_value(self.sigma, parameters)
        prediction_array = np.asarray(prediction, dtype=float)
        return np.square(prediction_array) * np.expm1(sigma * sigma) * np.exp(sigma * sigma)

    def logpdf(
        self,
        observed: ArrayLike,
        prediction: ArrayLike,
        parameters: Mapping[str, float] | None = None,
    ) -> NDArray[np.float64]:
        sigma = _positive("sigma", _parameter_value(self.sigma, parameters))
        observed_array = np.asarray(observed, dtype=float)
        prediction_array = np.asarray(prediction, dtype=float)
        if np.any(observed_array <= 0) or np.any(prediction_array <= 0):
            raise PKValidationError("log-normal observations and predictions must be positive")
        residual = (np.log(observed_array) - np.log(prediction_array)) / sigma
        return (
            -np.log(observed_array) - log(sigma) - 0.5 * log(2.0 * pi) - 0.5 * residual * residual
        )

    def simulate(
        self,
        prediction: ArrayLike,
        *,
        rng: np.random.Generator | None = None,
        parameters: Mapping[str, float] | None = None,
    ) -> NDArray[np.float64]:
        sigma = _parameter_value(self.sigma, parameters)
        prediction_array = np.asarray(prediction, dtype=float)
        generator = np.random.default_rng() if rng is None else rng
        return prediction_array * np.exp(generator.normal(0.0, sigma, prediction_array.shape))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "lognormal", "sigma": getattr(self.sigma, "name", self.sigma)}


def additive(sigma: ErrorParameter) -> AdditiveError:
    """Declare an additive Gaussian error model."""

    return AdditiveError(sigma)


def proportional(sigma: ErrorParameter) -> ProportionalError:
    """Declare a proportional Gaussian error model."""

    return ProportionalError(sigma)


def power(sigma: ErrorParameter, exponent: ErrorParameter) -> PowerError:
    """Declare a power residual-error model."""

    return PowerError(sigma, exponent)


def combined(
    additive_sigma: ErrorParameter,
    proportional_sigma: ErrorParameter,
    *,
    exponent: ErrorParameter = 1.0,
) -> CombinedError:
    """Declare an additive-plus-power residual-error model."""

    return CombinedError(additive_sigma, proportional_sigma, exponent)


def lognormal(sigma: ErrorParameter) -> LogNormalError:
    """Declare an original-scale log-normal error model."""

    return LogNormalError(sigma)


def left_censored_loglikelihood(
    limit: ArrayLike,
    prediction: ArrayLike,
    error: ObservationError,
    *,
    parameters: Mapping[str, float] | None = None,
) -> NDArray[np.float64]:
    """Stable log-CDF contribution for observations below ``limit``."""

    limit_array = np.asarray(limit, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    if isinstance(error, LogNormalError):
        sigma = _positive("sigma", _parameter_value(error.sigma, parameters))
        if np.any(limit_array <= 0) or np.any(prediction_array <= 0):
            raise PKValidationError("log-normal censoring limits and predictions must be positive")
        return log_ndtr((np.log(limit_array) - np.log(prediction_array)) / sigma)
    standard_deviation = np.sqrt(error.variance(prediction_array, parameters))
    return log_ndtr((limit_array - prediction_array) / standard_deviation)


def right_censored_loglikelihood(
    limit: ArrayLike,
    prediction: ArrayLike,
    error: ObservationError,
    *,
    parameters: Mapping[str, float] | None = None,
) -> NDArray[np.float64]:
    """Stable log-survival contribution for observations above ``limit``."""

    limit_array = np.asarray(limit, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    if isinstance(error, LogNormalError):
        sigma = _positive("sigma", _parameter_value(error.sigma, parameters))
        if np.any(limit_array <= 0) or np.any(prediction_array <= 0):
            raise PKValidationError("log-normal censoring limits and predictions must be positive")
        return log_ndtr((np.log(prediction_array) - np.log(limit_array)) / sigma)
    standard_deviation = np.sqrt(error.variance(prediction_array, parameters))
    return log_ndtr((prediction_array - limit_array) / standard_deviation)


def interval_censored_loglikelihood(
    lower: ArrayLike,
    upper: ArrayLike,
    prediction: ArrayLike,
    error: ObservationError,
    *,
    parameters: Mapping[str, float] | None = None,
) -> NDArray[np.float64]:
    """Stable log probability for an interval-censored observation."""

    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    if np.any(upper_array <= lower_array):
        raise PKValidationError("each censoring upper bound must exceed its lower bound")
    if isinstance(error, LogNormalError):
        sigma = _positive("sigma", _parameter_value(error.sigma, parameters))
        if np.any(lower_array <= 0) or np.any(upper_array <= 0) or np.any(prediction_array <= 0):
            raise PKValidationError("log-normal censoring bounds and predictions must be positive")
        z_lower = (np.log(lower_array) - np.log(prediction_array)) / sigma
        z_upper = (np.log(upper_array) - np.log(prediction_array)) / sigma
    else:
        standard_deviation = np.sqrt(error.variance(prediction_array, parameters))
        z_lower = (lower_array - prediction_array) / standard_deviation
        z_upper = (upper_array - prediction_array) / standard_deviation
    log_upper = log_ndtr(z_upper)
    log_lower = log_ndtr(z_lower)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = log_upper + np.log1p(-np.exp(log_lower - log_upper))
    return result


# Readable aliases retain the explicit IV names while accommodating common
# shorthand in notebooks and model libraries.
one_compartment_bolus = one_compartment_iv_bolus
one_compartment_iv_infusion = one_compartment_infusion
two_compartment_bolus = two_compartment_iv_bolus
two_compartment_iv_infusion = two_compartment_infusion


__all__ = [
    "AdditiveError",
    "CombinedError",
    "LogNormalError",
    "ObservationError",
    "OneCompartmentPK",
    "PKValidationError",
    "PowerError",
    "ProportionalError",
    "TwoCompartmentPK",
    "TwoCompartmentRates",
    "additive",
    "combined",
    "interval_censored_loglikelihood",
    "left_censored_loglikelihood",
    "lognormal",
    "one_compartment_bolus",
    "one_compartment_infusion",
    "one_compartment_iv_bolus",
    "one_compartment_iv_infusion",
    "one_compartment_oral",
    "power",
    "proportional",
    "right_censored_loglikelihood",
    "two_compartment_bolus",
    "two_compartment_infusion",
    "two_compartment_iv_bolus",
    "two_compartment_iv_infusion",
    "two_compartment_oral",
    "two_compartment_rates",
]
