"""Probability families and link functions used by PyMixEF.

The module follows one likelihood convention throughout: :meth:`Family.log_prob`
returns a *normalized* log density or log mass, including constants that depend
on the observation.  Consequently log likelihoods from these families can be
used for AIC and compared with other implementations after matching the stated
parameterization.

All random methods accept either a ``numpy.random.Generator`` or a seed through
``rng=``.  The implementations are NumPy/SciPy reference implementations; they
are intentionally independent of any automatic-differentiation framework.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, special, stats

_TINY = np.finfo(float).tiny
_EPS = np.finfo(float).eps


def _as_float(value: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(value, dtype=float)


def _return(value: ArrayLike) -> float | NDArray[np.float64]:
    result = np.asarray(value, dtype=float)
    return float(result) if result.ndim == 0 else result


def _generator(rng: np.random.Generator | int | None) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _probability(value: ArrayLike, name: str = "probability") -> NDArray[np.float64]:
    result = _as_float(value)
    if np.any(~np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _positive(value: ArrayLike, name: str) -> NDArray[np.float64]:
    result = _as_float(value)
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _integer_observation(value: ArrayLike, name: str = "observation") -> NDArray[np.float64]:
    result = _as_float(value)
    if np.any(~np.isfinite(result)) or np.any(result != np.floor(result)):
        raise ValueError(f"{name} must contain finite integers")
    return result


def _logdiffexp(log_a: ArrayLike, log_b: ArrayLike) -> NDArray[np.float64]:
    """Return log(exp(log_a) - exp(log_b)) for log_a >= log_b."""

    a, b = np.broadcast_arrays(_as_float(log_a), _as_float(log_b))
    out = np.full(a.shape, -np.inf, dtype=float)
    valid = a > b
    out[valid] = a[valid] + np.log1p(-np.exp(b[valid] - a[valid]))
    out[(a == b) & np.isneginf(a)] = -np.inf
    return out


@dataclass(frozen=True, slots=True)
class Link:
    """A differentiable scalar response link."""

    name: str
    _link: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    _inverse: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    _derivative: Callable[[NDArray[np.float64]], NDArray[np.float64]]

    def __call__(self, mean: ArrayLike) -> float | NDArray[np.float64]:
        return _return(self._link(_as_float(mean)))

    def inverse(self, linear_predictor: ArrayLike) -> float | NDArray[np.float64]:
        return _return(self._inverse(_as_float(linear_predictor)))

    def derivative(self, mean: ArrayLike) -> float | NDArray[np.float64]:
        """Derivative of the link with respect to the mean."""

        return _return(self._derivative(_as_float(mean)))


IDENTITY = Link("identity", lambda x: x, lambda x: x, lambda x: np.ones_like(x))
LOG = Link("log", np.log, np.exp, lambda x: 1.0 / x)
LOGIT = Link(
    "logit",
    lambda x: special.logit(x),
    lambda x: special.expit(x),
    lambda x: 1.0 / (x * (1.0 - x)),
)
PROBIT = Link(
    "probit",
    lambda x: special.ndtri(x),
    lambda x: special.ndtr(x),
    lambda x: 1.0 / np.maximum(stats.norm.pdf(special.ndtri(x)), _TINY),
)
CLOGLOG = Link(
    "cloglog",
    lambda x: np.log(-np.log1p(-x)),
    lambda x: -np.expm1(-np.exp(x)),
    lambda x: 1.0 / ((1.0 - x) * -np.log1p(-x)),
)
CAUCHIT = Link(
    "cauchit",
    lambda x: np.tan(np.pi * (x - 0.5)),
    lambda x: 0.5 + np.arctan(x) / np.pi,
    lambda x: np.pi / np.square(np.sin(np.pi * x)),
)
INVERSE = Link("inverse", lambda x: 1.0 / x, lambda x: 1.0 / x, lambda x: -1 / x**2)
INVERSE_SQUARED = Link(
    "inverse-squared",
    lambda x: 1.0 / x**2,
    lambda x: 1.0 / np.sqrt(x),
    lambda x: -2.0 / x**3,
)

_LINKS = {
    item.name: item
    for item in (IDENTITY, LOG, LOGIT, PROBIT, CLOGLOG, CAUCHIT, INVERSE, INVERSE_SQUARED)
}
_LINKS["inverse_squared"] = INVERSE_SQUARED


class links:
    """Namespace containing built-in :class:`Link` objects."""

    identity = IDENTITY
    log = LOG
    logit = LOGIT
    probit = PROBIT
    cloglog = CLOGLOG
    cauchit = CAUCHIT
    inverse = INVERSE
    inverse_squared = INVERSE_SQUARED


def get_link(value: str | Link | None, default: Link = IDENTITY) -> Link:
    """Resolve a link name or return an existing :class:`Link`."""

    if value is None:
        return default
    if isinstance(value, Link):
        return value
    try:
        return _LINKS[value.lower().replace(" ", "-")]
    except KeyError as exc:
        raise ValueError(f"unknown link {value!r}; available links are {sorted(_LINKS)}") from exc


class Family:
    """Base class for normalized probability families."""

    name = "family"
    support = "declared by subclass"
    parameter_names: tuple[str, ...] = ()
    default_link = IDENTITY
    discrete = False
    normalized = True

    def __init__(self, *, link: str | Link | None = None, **predictors: Any) -> None:
        self.link = get_link(link, self.default_link)
        self.predictors = dict(predictors)

    def log_prob(self, y: ArrayLike, **parameters: Any) -> float | NDArray[np.float64]:
        raise NotImplementedError

    def log_probability(self, y: ArrayLike, **parameters: Any) -> float | NDArray[np.float64]:
        """Long-form alias for :meth:`log_prob` used by plugin protocols."""

        return self.log_prob(y, **parameters)

    def logpdf(self, y: ArrayLike, **parameters: Any) -> float | NDArray[np.float64]:
        """SciPy-style alias; valid for both density and mass families."""

        return self.log_prob(y, **parameters)

    logpmf = logpdf

    def cdf(self, y: ArrayLike, **parameters: Any) -> float | NDArray[np.float64]:
        raise NotImplementedError(f"{self.name} does not provide a CDF")

    def logcdf(self, y: ArrayLike, **parameters: Any) -> float | NDArray[np.float64]:
        with np.errstate(divide="ignore"):
            return _return(np.log(self.cdf(y, **parameters)))

    def sf(self, y: ArrayLike, **parameters: Any) -> float | NDArray[np.float64]:
        return _return(1.0 - _as_float(self.cdf(y, **parameters)))

    def logsf(self, y: ArrayLike, **parameters: Any) -> float | NDArray[np.float64]:
        with np.errstate(divide="ignore"):
            return _return(np.log1p(-_as_float(self.cdf(y, **parameters))))

    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        *,
        rng: np.random.Generator | int | None = None,
        **parameters: Any,
    ) -> NDArray[Any] | np.generic:
        raise NotImplementedError

    def mean(self, **parameters: Any) -> float | NDArray[np.float64]:
        raise NotImplementedError

    def variance(self, **parameters: Any) -> float | NDArray[np.float64]:
        raise NotImplementedError

    def moments(self, **parameters: Any) -> dict[str, float | NDArray[np.float64]]:
        """Return the common first two moments."""

        return {"mean": self.mean(**parameters), "variance": self.variance(**parameters)}

    def random(
        self,
        size: int | tuple[int, ...] | None = None,
        *,
        rng: np.random.Generator | int | None = None,
        **parameters: Any,
    ) -> NDArray[Any] | np.generic:
        """Alias for :meth:`rvs`."""

        return self.rvs(size=size, rng=rng, **parameters)

    simulate = random


class Gaussian(Family):
    """Gaussian ``N(mu, sigma**2)`` family."""

    name = "gaussian"
    support = "real"
    parameter_names = ("mu", "sigma")
    default_link = IDENTITY

    def log_prob(self, y: ArrayLike, *, mu: ArrayLike = 0.0, sigma: ArrayLike = 1.0, **_: Any):
        sigma = _positive(sigma, "sigma")
        return _return(stats.norm.logpdf(_as_float(y), loc=mu, scale=sigma))

    def cdf(self, y: ArrayLike, *, mu: ArrayLike = 0.0, sigma: ArrayLike = 1.0, **_: Any):
        return _return(stats.norm.cdf(_as_float(y), loc=mu, scale=_positive(sigma, "sigma")))

    def logcdf(self, y: ArrayLike, *, mu: ArrayLike = 0.0, sigma: ArrayLike = 1.0, **_: Any):
        return _return(stats.norm.logcdf(_as_float(y), loc=mu, scale=_positive(sigma, "sigma")))

    def logsf(self, y: ArrayLike, *, mu: ArrayLike = 0.0, sigma: ArrayLike = 1.0, **_: Any):
        return _return(stats.norm.logsf(_as_float(y), loc=mu, scale=_positive(sigma, "sigma")))

    def rvs(self, size=None, *, rng=None, mu=0.0, sigma=1.0, **_: Any):
        return _generator(rng).normal(mu, _positive(sigma, "sigma"), size=size)

    def mean(self, *, mu=0.0, **_: Any):
        return _return(mu)

    def variance(self, *, sigma=1.0, **_: Any):
        return _return(_positive(sigma, "sigma") ** 2)


class StudentT(Family):
    """Student-t location-scale family with degrees of freedom ``df``."""

    name = "student-t"
    support = "real"
    parameter_names = ("mu", "sigma", "df")
    default_link = IDENTITY

    def __init__(self, df: float | None = None, *, link=None, **predictors: Any) -> None:
        super().__init__(link=link, **predictors)
        self.df = df

    def _df(self, df: ArrayLike | None) -> NDArray[np.float64]:
        return _positive(self.df if df is None else df, "df")

    def log_prob(self, y, *, mu=0.0, sigma=1.0, df=None, **_: Any):
        return _return(
            stats.t.logpdf(_as_float(y), df=self._df(df), loc=mu, scale=_positive(sigma, "sigma"))
        )

    def cdf(self, y, *, mu=0.0, sigma=1.0, df=None, **_: Any):
        return _return(
            stats.t.cdf(_as_float(y), df=self._df(df), loc=mu, scale=_positive(sigma, "sigma"))
        )

    def logcdf(self, y, *, mu=0.0, sigma=1.0, df=None, **_: Any):
        return _return(
            stats.t.logcdf(_as_float(y), df=self._df(df), loc=mu, scale=_positive(sigma, "sigma"))
        )

    def logsf(self, y, *, mu=0.0, sigma=1.0, df=None, **_: Any):
        return _return(
            stats.t.logsf(_as_float(y), df=self._df(df), loc=mu, scale=_positive(sigma, "sigma"))
        )

    def rvs(self, size=None, *, rng=None, mu=0.0, sigma=1.0, df=None, **_: Any):
        return stats.t.rvs(
            self._df(df),
            loc=mu,
            scale=_positive(sigma, "sigma"),
            size=size,
            random_state=_generator(rng),
        )

    def mean(self, *, mu=0.0, df=None, **_: Any):
        df_value = self._df(df)
        return _return(np.where(df_value > 1, mu, np.nan))

    def variance(self, *, sigma=1.0, df=None, **_: Any):
        df_value = self._df(df)
        result = np.where(df_value > 2, sigma**2 * df_value / (df_value - 2), np.inf)
        result = np.where(df_value <= 1, np.nan, result)
        return _return(result)


class LogNormal(Family):
    """Lognormal family parameterized by log-median ``meanlog`` and ``sigma``.

    ``exp(meanlog)`` is the median.  The arithmetic mean is
    ``exp(meanlog + sigma**2 / 2)``.
    """

    name = "lognormal"
    support = "positive real"
    parameter_names = ("meanlog", "sigma")
    default_link = LOG

    def log_prob(self, y, *, meanlog=0.0, sigma=1.0, mu=None, **_: Any):
        meanlog = meanlog if mu is None else mu
        return _return(
            stats.lognorm.logpdf(_as_float(y), s=_positive(sigma, "sigma"), scale=np.exp(meanlog))
        )

    def cdf(self, y, *, meanlog=0.0, sigma=1.0, mu=None, **_: Any):
        meanlog = meanlog if mu is None else mu
        return _return(
            stats.lognorm.cdf(_as_float(y), s=_positive(sigma, "sigma"), scale=np.exp(meanlog))
        )

    def logcdf(self, y, *, meanlog=0.0, sigma=1.0, mu=None, **_: Any):
        meanlog = meanlog if mu is None else mu
        return _return(
            stats.lognorm.logcdf(_as_float(y), s=_positive(sigma, "sigma"), scale=np.exp(meanlog))
        )

    def logsf(self, y, *, meanlog=0.0, sigma=1.0, mu=None, **_: Any):
        meanlog = meanlog if mu is None else mu
        return _return(
            stats.lognorm.logsf(_as_float(y), s=_positive(sigma, "sigma"), scale=np.exp(meanlog))
        )

    def rvs(self, size=None, *, rng=None, meanlog=0.0, sigma=1.0, mu=None, **_: Any):
        meanlog = meanlog if mu is None else mu
        return _generator(rng).lognormal(meanlog, _positive(sigma, "sigma"), size=size)

    def mean(self, *, meanlog=0.0, sigma=1.0, mu=None, **_: Any):
        meanlog = meanlog if mu is None else mu
        return _return(np.exp(_as_float(meanlog) + _positive(sigma, "sigma") ** 2 / 2))

    def variance(self, *, meanlog=0.0, sigma=1.0, mu=None, **_: Any):
        meanlog = meanlog if mu is None else mu
        s2 = _positive(sigma, "sigma") ** 2
        return _return(np.expm1(s2) * np.exp(2 * _as_float(meanlog) + s2))


class Gamma(Family):
    """Gamma family in mean-dispersion form.

    ``E(Y)=mu`` and ``Var(Y)=dispersion * mu**2``.  Thus shape is
    ``1 / dispersion`` and scale is ``mu * dispersion``.
    """

    name = "gamma"
    support = "positive real"
    parameter_names = ("mu", "dispersion")
    default_link = LOG

    @staticmethod
    def _shape_scale(
        mu: ArrayLike, dispersion: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        mu = _positive(mu, "mu")
        dispersion = _positive(dispersion, "dispersion")
        return 1.0 / dispersion, mu * dispersion

    def log_prob(self, y, *, mu=1.0, dispersion=1.0, **_: Any):
        shape, scale = self._shape_scale(mu, dispersion)
        return _return(stats.gamma.logpdf(_as_float(y), a=shape, scale=scale))

    def cdf(self, y, *, mu=1.0, dispersion=1.0, **_: Any):
        shape, scale = self._shape_scale(mu, dispersion)
        return _return(stats.gamma.cdf(_as_float(y), a=shape, scale=scale))

    def logcdf(self, y, *, mu=1.0, dispersion=1.0, **_: Any):
        shape, scale = self._shape_scale(mu, dispersion)
        return _return(stats.gamma.logcdf(_as_float(y), a=shape, scale=scale))

    def logsf(self, y, *, mu=1.0, dispersion=1.0, **_: Any):
        shape, scale = self._shape_scale(mu, dispersion)
        return _return(stats.gamma.logsf(_as_float(y), a=shape, scale=scale))

    def rvs(self, size=None, *, rng=None, mu=1.0, dispersion=1.0, **_: Any):
        shape, scale = self._shape_scale(mu, dispersion)
        return _generator(rng).gamma(shape, scale, size=size)

    def mean(self, *, mu=1.0, **_: Any):
        return _return(mu)

    def variance(self, *, mu=1.0, dispersion=1.0, **_: Any):
        return _return(_positive(dispersion, "dispersion") * _positive(mu, "mu") ** 2)


class InverseGaussian(Family):
    """Inverse-Gaussian family with ``E(Y)=mu`` and ``Var(Y)=phi*mu**3``."""

    name = "inverse-gaussian"
    support = "positive real"
    parameter_names = ("mu", "dispersion")
    default_link = LOG

    @staticmethod
    def _scipy(mu: ArrayLike, dispersion: ArrayLike):
        mu = _positive(mu, "mu")
        phi = _positive(dispersion, "dispersion")
        return mu * phi, 1.0 / phi

    def log_prob(self, y, *, mu=1.0, dispersion=1.0, **_: Any):
        shape, scale = self._scipy(mu, dispersion)
        return _return(stats.invgauss.logpdf(_as_float(y), shape, scale=scale))

    def cdf(self, y, *, mu=1.0, dispersion=1.0, **_: Any):
        shape, scale = self._scipy(mu, dispersion)
        return _return(stats.invgauss.cdf(_as_float(y), shape, scale=scale))

    def logcdf(self, y, *, mu=1.0, dispersion=1.0, **_: Any):
        shape, scale = self._scipy(mu, dispersion)
        return _return(stats.invgauss.logcdf(_as_float(y), shape, scale=scale))

    def logsf(self, y, *, mu=1.0, dispersion=1.0, **_: Any):
        shape, scale = self._scipy(mu, dispersion)
        return _return(stats.invgauss.logsf(_as_float(y), shape, scale=scale))

    def rvs(self, size=None, *, rng=None, mu=1.0, dispersion=1.0, **_: Any):
        shape, scale = self._scipy(mu, dispersion)
        return stats.invgauss.rvs(shape, scale=scale, size=size, random_state=_generator(rng))

    def mean(self, *, mu=1.0, **_: Any):
        return _return(mu)

    def variance(self, *, mu=1.0, dispersion=1.0, **_: Any):
        return _return(_positive(dispersion, "dispersion") * _positive(mu, "mu") ** 3)


class Beta(Family):
    """Beta family parameterized by mean ``mu`` and precision ``precision``."""

    name = "beta"
    support = "open unit interval"
    parameter_names = ("mu", "precision")
    default_link = LOGIT

    @staticmethod
    def _shapes(mu: ArrayLike, precision: ArrayLike):
        mu = _probability(mu, "mu")
        if np.any((mu == 0) | (mu == 1)):
            raise ValueError("mu must be strictly between zero and one")
        precision = _positive(precision, "precision")
        return mu * precision, (1.0 - mu) * precision

    def log_prob(self, y, *, mu=0.5, precision=2.0, **_: Any):
        alpha, beta = self._shapes(mu, precision)
        return _return(stats.beta.logpdf(_as_float(y), alpha, beta))

    def cdf(self, y, *, mu=0.5, precision=2.0, **_: Any):
        alpha, beta = self._shapes(mu, precision)
        return _return(stats.beta.cdf(_as_float(y), alpha, beta))

    def logcdf(self, y, *, mu=0.5, precision=2.0, **_: Any):
        alpha, beta = self._shapes(mu, precision)
        return _return(stats.beta.logcdf(_as_float(y), alpha, beta))

    def logsf(self, y, *, mu=0.5, precision=2.0, **_: Any):
        alpha, beta = self._shapes(mu, precision)
        return _return(stats.beta.logsf(_as_float(y), alpha, beta))

    def rvs(self, size=None, *, rng=None, mu=0.5, precision=2.0, **_: Any):
        alpha, beta = self._shapes(mu, precision)
        return _generator(rng).beta(alpha, beta, size=size)

    def mean(self, *, mu=0.5, **_: Any):
        return _return(mu)

    def variance(self, *, mu=0.5, precision=2.0, **_: Any):
        mu = _probability(mu, "mu")
        return _return(mu * (1 - mu) / (_positive(precision, "precision") + 1))


class Bernoulli(Family):
    """Bernoulli family with success probability ``mu``."""

    name = "bernoulli"
    support = "{0, 1}"
    parameter_names = ("mu",)
    default_link = LOGIT
    discrete = True

    def log_prob(self, y, *, mu=0.5, **_: Any):
        y = _integer_observation(y)
        mu = _probability(mu, "mu")
        out = stats.bernoulli.logpmf(y, mu)
        return _return(out)

    def cdf(self, y, *, mu=0.5, **_: Any):
        return _return(stats.bernoulli.cdf(_as_float(y), _probability(mu, "mu")))

    def rvs(self, size=None, *, rng=None, mu=0.5, **_: Any):
        return _generator(rng).binomial(1, _probability(mu, "mu"), size=size)

    def mean(self, *, mu=0.5, **_: Any):
        return _return(_probability(mu, "mu"))

    def variance(self, *, mu=0.5, **_: Any):
        mu = _probability(mu, "mu")
        return _return(mu * (1 - mu))


class Binomial(Family):
    """Binomial family; observations are successes and ``trials`` is explicit."""

    name = "binomial"
    support = "integers from zero through trials"
    parameter_names = ("mu", "trials")
    default_link = LOGIT
    discrete = True

    def __init__(self, trials: ArrayLike | None = None, *, link=None, **predictors: Any) -> None:
        super().__init__(link=link, **predictors)
        self.trials = trials

    def _trials(self, trials: ArrayLike | None) -> NDArray[np.float64]:
        if trials is None:
            trials = self.trials
        if trials is None:
            raise ValueError("Binomial requires trials= at construction or evaluation")
        result = _integer_observation(trials, "trials")
        if np.any(result < 0):
            raise ValueError("trials must be nonnegative")
        return result

    def log_prob(self, y, *, mu=0.5, trials=None, **_: Any):
        y = _integer_observation(y)
        return _return(stats.binom.logpmf(y, self._trials(trials), _probability(mu, "mu")))

    def cdf(self, y, *, mu=0.5, trials=None, **_: Any):
        return _return(stats.binom.cdf(_as_float(y), self._trials(trials), _probability(mu, "mu")))

    def rvs(self, size=None, *, rng=None, mu=0.5, trials=None, **_: Any):
        return _generator(rng).binomial(self._trials(trials), _probability(mu, "mu"), size=size)

    def mean(self, *, mu=0.5, trials=None, **_: Any):
        return _return(self._trials(trials) * _probability(mu, "mu"))

    def variance(self, *, mu=0.5, trials=None, **_: Any):
        mu = _probability(mu, "mu")
        return _return(self._trials(trials) * mu * (1 - mu))


class Poisson(Family):
    """Poisson family with mean/rate ``mu``."""

    name = "poisson"
    support = "nonnegative integers"
    parameter_names = ("mu",)
    default_link = LOG
    discrete = True

    def log_prob(self, y, *, mu=1.0, **_: Any):
        return _return(stats.poisson.logpmf(_integer_observation(y), _positive(mu, "mu")))

    def cdf(self, y, *, mu=1.0, **_: Any):
        return _return(stats.poisson.cdf(_as_float(y), _positive(mu, "mu")))

    def rvs(self, size=None, *, rng=None, mu=1.0, **_: Any):
        return _generator(rng).poisson(_positive(mu, "mu"), size=size)

    def mean(self, *, mu=1.0, **_: Any):
        return _return(_positive(mu, "mu"))

    def variance(self, *, mu=1.0, **_: Any):
        return _return(_positive(mu, "mu"))


class NegativeBinomial2(Family):
    """NB2 family with ``Var(Y)=mu + mu**2 / dispersion``.

    ``dispersion`` is the conventional positive size/shape parameter.
    """

    name = "negative-binomial-2"
    support = "nonnegative integers"
    parameter_names = ("mu", "dispersion")
    default_link = LOG
    discrete = True

    def __init__(self, dispersion: Any = None, *, link=None, **predictors: Any) -> None:
        if isinstance(dispersion, str):
            predictors["dispersion"] = dispersion
            dispersion = None
        super().__init__(link=link, **predictors)
        self.fixed_dispersion = dispersion

    def _dispersion(self, value: ArrayLike | None) -> NDArray[np.float64]:
        value = self.fixed_dispersion if value is None else value
        return _positive(1.0 if value is None else value, "dispersion")

    def _params(self, mu, dispersion):
        size = self._dispersion(dispersion)
        mu = _positive(mu, "mu")
        return size, size / (size + mu)

    def log_prob(self, y, *, mu=1.0, dispersion=None, **_: Any):
        size, probability = self._params(mu, dispersion)
        return _return(stats.nbinom.logpmf(_integer_observation(y), size, probability))

    def cdf(self, y, *, mu=1.0, dispersion=None, **_: Any):
        size, probability = self._params(mu, dispersion)
        return _return(stats.nbinom.cdf(_as_float(y), size, probability))

    def rvs(self, size=None, *, rng=None, mu=1.0, dispersion=None, **_: Any):
        shape, probability = self._params(mu, dispersion)
        return _generator(rng).negative_binomial(shape, probability, size=size)

    def mean(self, *, mu=1.0, **_: Any):
        return _return(_positive(mu, "mu"))

    def variance(self, *, mu=1.0, dispersion=None, **_: Any):
        mu = _positive(mu, "mu")
        return _return(mu + mu**2 / self._dispersion(dispersion))


class NegativeBinomial1(NegativeBinomial2):
    """NB1 family with ``Var(Y)=mu * (1 + dispersion)``."""

    name = "negative-binomial-1"

    def _params(self, mu, dispersion):
        alpha = self._dispersion(dispersion)
        mu = _positive(mu, "mu")
        size = mu / alpha
        return size, 1.0 / (1.0 + alpha)

    def variance(self, *, mu=1.0, dispersion=None, **_: Any):
        return _return(_positive(mu, "mu") * (1.0 + self._dispersion(dispersion)))


class GeneralizedPoisson(Family):
    """Consul--Jain generalized Poisson in mean-dispersion form.

    The underlying parameters are ``lambda = mu * (1-dispersion)`` and
    ``theta = dispersion``.  The mass is
    ``lambda * (lambda + theta*y)**(y-1) * exp(-(lambda+theta*y))/y!``.
    This gives arithmetic mean ``mu`` for the regular infinite-support case
    ``0 <= theta < 1``.  Negative theta has finite support.
    """

    name = "generalized-poisson"
    support = "nonnegative integers subject to lambda + dispersion*y > 0"
    parameter_names = ("mu", "dispersion")
    default_link = LOG
    discrete = True

    @staticmethod
    def _base(mu, dispersion):
        mu = _positive(mu, "mu")
        theta = _as_float(dispersion)
        if np.any(~np.isfinite(theta)) or np.any(theta >= 1):
            raise ValueError("generalized-Poisson dispersion must be finite and less than one")
        lam = mu * (1 - theta)
        return lam, theta

    def log_prob(self, y, *, mu=1.0, dispersion=0.0, **_: Any):
        y = _integer_observation(y)
        lam, theta = self._base(mu, dispersion)
        y, lam, theta = np.broadcast_arrays(y, lam, theta)
        term = lam + theta * y
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.log(lam) + (y - 1) * np.log(term) - term - special.gammaln(y + 1)
        out = np.where((y >= 0) & (term > 0), out, -np.inf)
        out = np.where((y == 0) & (lam > 0), -lam, out)
        return _return(out)

    def cdf(self, y, *, mu=1.0, dispersion=0.0, **parameters: Any):
        values = np.floor(_as_float(y)).astype(int)
        mu_a, theta_a, values = np.broadcast_arrays(_as_float(mu), _as_float(dispersion), values)
        out = np.empty(values.shape)
        for index in np.ndindex(values.shape):
            top = values[index]
            if top < 0:
                out[index] = 0.0
            else:
                grid = np.arange(top + 1)
                out[index] = np.exp(
                    self.log_prob(grid, mu=mu_a[index], dispersion=theta_a[index])
                ).sum()
        return _return(np.minimum(out, 1.0))

    def rvs(self, size=None, *, rng=None, mu=1.0, dispersion=0.0, **_: Any):
        generator = _generator(rng)
        shape = () if size is None else ((size,) if isinstance(size, int) else size)
        mu_b, theta_b = np.broadcast_arrays(_as_float(mu), _as_float(dispersion))
        output_shape = shape + mu_b.shape
        uniforms = generator.random(output_shape)
        out = np.empty(output_shape, dtype=int)
        for index in np.ndindex(output_shape):
            parameter_index = index[len(shape) :] if mu_b.ndim else ()
            m = float(mu_b[parameter_index])
            t = float(theta_b[parameter_index])
            u, cumulative, k = uniforms[index], 0.0, 0
            while cumulative < u and k < 100_000:
                cumulative += float(np.exp(self.log_prob(k, mu=m, dispersion=t)))
                if cumulative >= u:
                    break
                k += 1
                if t < 0 and m * (1 - t) + t * k <= 0:
                    break
            out[index] = k
        return out.item() if out.ndim == 0 else out

    def mean(self, *, mu=1.0, **_: Any):
        return _return(_positive(mu, "mu"))

    def variance(self, *, mu=1.0, dispersion=0.0, **_: Any):
        theta = _as_float(dispersion)
        if np.any(theta >= 1):
            raise ValueError("dispersion must be less than one")
        return _return(_positive(mu, "mu") / (1 - theta) ** 2)


class COMPoisson(Family):
    """Conway--Maxwell--Poisson with rate ``rate`` and exponent ``dispersion``.

    ``P(Y=y) = rate**y / (y!)**dispersion / Z(rate, dispersion)``.
    The ``rate`` parameter is *not* generally the arithmetic mean.  Normalizing
    constants are evaluated by an adaptive log-series and an error is raised if
    the configured term limit is insufficient.
    """

    name = "com-poisson"
    support = "nonnegative integers"
    parameter_names = ("rate", "dispersion")
    default_link = LOG
    discrete = True

    def __init__(
        self, *, link=None, max_terms: int = 100_000, tolerance: float = 1e-13, **predictors
    ):
        super().__init__(link=link, **predictors)
        self.max_terms = int(max_terms)
        self.tolerance = float(tolerance)

    def _log_terms(self, rate: float, dispersion: float) -> NDArray[np.float64]:
        if not np.isfinite(rate) or rate <= 0 or not np.isfinite(dispersion) or dispersion <= 0:
            raise ValueError("COM-Poisson rate and dispersion must be finite and positive")
        logs = [0.0]
        current = 0.0
        log_rate = np.log(rate)
        peak_seen = False
        for k in range(1, self.max_terms + 1):
            current += log_rate - dispersion * np.log(k)
            logs.append(current)
            peak_seen = peak_seen or current < logs[-2]
            if peak_seen and current - max(logs) < np.log(self.tolerance):
                return np.asarray(logs)
        raise RuntimeError(
            "COM-Poisson normalization did not converge; increase max_terms or rescale parameters"
        )

    def _logz_scalar(self, rate: float, dispersion: float) -> float:
        return float(special.logsumexp(self._log_terms(rate, dispersion)))

    def _broadcast_logz(self, rate, dispersion):
        rate, dispersion = np.broadcast_arrays(
            _positive(rate, "rate"), _positive(dispersion, "dispersion")
        )
        out = np.empty(rate.shape)
        for index in np.ndindex(rate.shape):
            out[index] = self._logz_scalar(float(rate[index]), float(dispersion[index]))
        return out

    def log_prob(self, y, *, rate=None, mu=None, dispersion=1.0, **_: Any):
        rate = mu if rate is None else rate
        if rate is None:
            rate = 1.0
        y = _integer_observation(y)
        rate, dispersion, y = np.broadcast_arrays(
            _positive(rate, "rate"), _positive(dispersion, "dispersion"), y
        )
        logz = self._broadcast_logz(rate, dispersion)
        out = y * np.log(rate) - dispersion * special.gammaln(y + 1) - logz
        return _return(np.where(y >= 0, out, -np.inf))

    def cdf(self, y, *, rate=None, mu=None, dispersion=1.0, **_: Any):
        rate = mu if rate is None else rate
        if rate is None:
            rate = 1.0
        y, rate, dispersion = np.broadcast_arrays(
            np.floor(_as_float(y)).astype(int),
            _positive(rate, "rate"),
            _positive(dispersion, "dispersion"),
        )
        out = np.empty(y.shape)
        for index in np.ndindex(y.shape):
            if y[index] < 0:
                out[index] = 0.0
                continue
            terms = self._log_terms(float(rate[index]), float(dispersion[index]))
            top = min(int(y[index]), len(terms) - 1)
            out[index] = np.exp(special.logsumexp(terms[: top + 1]) - special.logsumexp(terms))
        return _return(out)

    def rvs(self, size=None, *, rng=None, rate=None, mu=None, dispersion=1.0, **_: Any):
        rate = mu if rate is None else rate
        if rate is None:
            rate = 1.0
        rate_a, disp_a = np.broadcast_arrays(
            _positive(rate, "rate"), _positive(dispersion, "dispersion")
        )
        sample_shape = () if size is None else ((size,) if isinstance(size, int) else size)
        out = np.empty(sample_shape + rate_a.shape, dtype=int)
        generator = _generator(rng)
        for index in np.ndindex(rate_a.shape):
            terms = self._log_terms(float(rate_a[index]), float(disp_a[index]))
            probabilities = np.exp(terms - special.logsumexp(terms))
            draws = generator.choice(len(terms), size=sample_shape or None, p=probabilities)
            out[(...,) + index] = draws
        return out.item() if out.ndim == 0 else out

    def mean(self, *, rate=None, mu=None, dispersion=1.0, **_: Any):
        rate = mu if rate is None else rate
        if rate is None:
            rate = 1.0
        rate_a, disp_a = np.broadcast_arrays(
            _positive(rate, "rate"), _positive(dispersion, "dispersion")
        )
        out = np.empty(rate_a.shape)
        for index in np.ndindex(rate_a.shape):
            terms = self._log_terms(float(rate_a[index]), float(disp_a[index]))
            p = np.exp(terms - special.logsumexp(terms))
            out[index] = np.dot(np.arange(len(p)), p)
        return _return(out)

    def variance(self, *, rate=None, mu=None, dispersion=1.0, **_: Any):
        rate = mu if rate is None else rate
        if rate is None:
            rate = 1.0
        rate_a, disp_a = np.broadcast_arrays(
            _positive(rate, "rate"), _positive(dispersion, "dispersion")
        )
        out = np.empty(rate_a.shape)
        for index in np.ndindex(rate_a.shape):
            terms = self._log_terms(float(rate_a[index]), float(disp_a[index]))
            p = np.exp(terms - special.logsumexp(terms))
            grid = np.arange(len(p))
            mean = np.dot(grid, p)
            out[index] = np.dot((grid - mean) ** 2, p)
        return _return(out)


class Tweedie(Family):
    """Tweedie exponential-dispersion family.

    Exact normalized reference likelihood and RNG are implemented for
    ``1 < power < 2`` (compound Poisson--Gamma), and for the canonical limits
    power 0 (Gaussian), 1 (Poisson), 2 (Gamma), and 3 (inverse Gaussian).
    Other powers are rejected rather than evaluated with an unnormalized
    quasi-likelihood.  Compound-series evaluation is intentionally capped; very
    large Poisson intensities raise a stable ``RuntimeError``.
    """

    name = "tweedie"
    support = "depends on power"
    parameter_names = ("mu", "dispersion", "power")
    default_link = LOG

    def __init__(self, power: float = 1.5, *, link=None, max_terms: int = 100_000, **predictors):
        super().__init__(link=link, **predictors)
        self.power = float(power)
        self.max_terms = int(max_terms)

    def _parameters(self, mu, dispersion, power):
        p = self.power if power is None else power
        mu_value = _as_float(mu)
        power_value = _as_float(p)
        if np.any(~np.isfinite(mu_value)):
            raise ValueError("mu must be finite")
        mu_broadcast, power_broadcast = np.broadcast_arrays(mu_value, power_value)
        if np.any((~np.isclose(power_broadcast, 0)) & (mu_broadcast <= 0)):
            raise ValueError("Tweedie mu must be positive except at the Gaussian power=0 limit")
        return mu_value, _positive(dispersion, "dispersion"), power_value

    def _compound_logpdf_scalar(self, y: float, mu: float, phi: float, power: float) -> float:
        lam = mu ** (2 - power) / (phi * (2 - power))
        if y < 0:
            return -np.inf
        if y == 0:
            return -lam
        if lam > 10_000:
            raise RuntimeError(
                "Tweedie compound-Poisson intensity exceeds reference-series limit (10000)"
            )
        alpha = (2 - power) / (power - 1)
        scale = phi * (power - 1) * mu ** (power - 1)
        top = max(50, int(np.ceil(lam + 15 * np.sqrt(lam + 1) + 50)))
        top = min(top, self.max_terms)
        j = np.arange(1, top + 1, dtype=float)
        terms = stats.poisson.logpmf(j, lam) + stats.gamma.logpdf(y, a=j * alpha, scale=scale)
        if top == self.max_terms and terms[-1] > np.max(terms) - 30:
            raise RuntimeError("Tweedie density series did not converge within max_terms")
        return float(special.logsumexp(terms))

    def log_prob(self, y, *, mu=1.0, dispersion=1.0, power=None, **_: Any):
        mu, phi, power = self._parameters(mu, dispersion, power)
        y, mu, phi, power = np.broadcast_arrays(_as_float(y), mu, phi, power)
        out = np.empty(y.shape)
        for index in np.ndindex(y.shape):
            yi, mui, phii, pi = map(float, (y[index], mu[index], phi[index], power[index]))
            if np.isclose(pi, 0):
                out[index] = stats.norm.logpdf(yi, mui, np.sqrt(phii))
            elif np.isclose(pi, 1):
                out[index] = stats.poisson.logpmf(yi, mui / phii)
                # EDM scaling at phi != 1 is not a Poisson probability law.
                if not np.isclose(phii, 1):
                    raise ValueError("Tweedie power=1 is a Poisson law only when dispersion=1")
            elif 1 < pi < 2:
                out[index] = self._compound_logpdf_scalar(yi, mui, phii, pi)
            elif np.isclose(pi, 2):
                out[index] = Gamma().log_prob(yi, mu=mui, dispersion=phii)
            elif np.isclose(pi, 3):
                out[index] = InverseGaussian().log_prob(yi, mu=mui, dispersion=phii)
            else:
                raise NotImplementedError(
                    "normalized Tweedie density is implemented only for 1<p<2 and powers 0,1,2,3"
                )
        return _return(out)

    def cdf(self, y, *, mu=1.0, dispersion=1.0, power=None, **_: Any):
        mu, phi, power = self._parameters(mu, dispersion, power)
        y, mu, phi, power = np.broadcast_arrays(_as_float(y), mu, phi, power)
        out = np.empty(y.shape)
        for index in np.ndindex(y.shape):
            yi, mui, phii, pi = map(float, (y[index], mu[index], phi[index], power[index]))
            if np.isclose(pi, 0):
                out[index] = stats.norm.cdf(yi, mui, np.sqrt(phii))
            elif np.isclose(pi, 1) and np.isclose(phii, 1):
                out[index] = stats.poisson.cdf(yi, mui)
            elif np.isclose(pi, 2):
                out[index] = Gamma().cdf(yi, mu=mui, dispersion=phii)
            elif np.isclose(pi, 3):
                out[index] = InverseGaussian().cdf(yi, mu=mui, dispersion=phii)
            else:
                raise NotImplementedError("Tweedie CDF is unavailable for compound powers 1<p<2")
        return _return(out)

    def rvs(self, size=None, *, rng=None, mu=1.0, dispersion=1.0, power=None, **_: Any):
        mu, phi, p = self._parameters(mu, dispersion, power)
        generator = _generator(rng)
        if np.ndim(p) != 0:
            raise ValueError("Tweedie rvs currently requires a scalar power")
        p_value = float(p)
        if np.isclose(p_value, 0):
            return generator.normal(mu, np.sqrt(phi), size=size)
        if np.isclose(p_value, 1):
            if np.any(~np.isclose(phi, 1)):
                raise ValueError("Tweedie power=1 is a Poisson law only when dispersion=1")
            return generator.poisson(mu, size=size)
        if 1 < p_value < 2:
            lam = mu ** (2 - p_value) / (phi * (2 - p_value))
            alpha = (2 - p_value) / (p_value - 1)
            scale = phi * (p_value - 1) * mu ** (p_value - 1)
            counts = generator.poisson(lam, size=size)
            return generator.gamma(counts * alpha, scale)
        if np.isclose(p_value, 2):
            return Gamma().rvs(size, rng=generator, mu=mu, dispersion=phi)
        if np.isclose(p_value, 3):
            return InverseGaussian().rvs(size, rng=generator, mu=mu, dispersion=phi)
        raise NotImplementedError("Tweedie RNG is implemented only for 1<p<2 and powers 0,1,2,3")

    def mean(self, *, mu=1.0, **_: Any):
        return _return(_positive(mu, "mu"))

    def variance(self, *, mu=1.0, dispersion=1.0, power=None, **_: Any):
        p = self.power if power is None else power
        return _return(_positive(dispersion, "dispersion") * _positive(mu, "mu") ** _as_float(p))


class Ordinal(Family):
    """Cumulative-link ordinal family with ordered zero-based categories."""

    name = "ordinal"
    support = "ordered categories 0, ..., len(thresholds)"
    parameter_names = ("eta", "thresholds")
    default_link = LOGIT
    discrete = True

    def __init__(
        self, *, link: str | Link | None = None, thresholds: ArrayLike | None = None, **predictors
    ):
        super().__init__(link=link or LOGIT, **predictors)
        if self.link.name not in {"logit", "probit", "cloglog"}:
            raise ValueError("Ordinal supports cumulative logit, probit, or cloglog links")
        self.thresholds = None if thresholds is None else _as_float(thresholds)

    def _thresholds(self, thresholds):
        result = self.thresholds if thresholds is None else _as_float(thresholds)
        if result is None or result.ndim != 1 or result.size == 0:
            raise ValueError("Ordinal requires a one-dimensional, nonempty thresholds array")
        if np.any(~np.isfinite(result)) or np.any(np.diff(result) <= 0):
            raise ValueError("ordinal thresholds must be finite and strictly increasing")
        return result

    def _latent_cdf(self, value):
        return _as_float(self.link.inverse(value))

    def probabilities(self, *, eta=0.0, thresholds=None):
        thresholds = self._thresholds(thresholds)
        eta = _as_float(eta)
        cut = self._latent_cdf(thresholds - eta[..., None])
        return np.diff(
            np.concatenate([np.zeros(eta.shape + (1,)), cut, np.ones(eta.shape + (1,))], axis=-1),
            axis=-1,
        )

    def log_prob(self, y, *, eta=0.0, thresholds=None, **_: Any):
        y = _integer_observation(y).astype(int)
        probabilities = self.probabilities(eta=eta, thresholds=thresholds)
        shape = np.broadcast_shapes(y.shape, probabilities.shape[:-1])
        y = np.broadcast_to(y, shape)
        probabilities = np.broadcast_to(probabilities, shape + (probabilities.shape[-1],))
        valid = (y >= 0) & (y < probabilities.shape[-1])
        clipped = np.clip(y, 0, probabilities.shape[-1] - 1)
        selected = np.take_along_axis(probabilities, clipped[..., None], axis=-1)[..., 0]
        with np.errstate(divide="ignore"):
            return _return(np.where(valid, np.log(selected), -np.inf))

    def cdf(self, y, *, eta=0.0, thresholds=None, **_: Any):
        y = np.floor(_as_float(y)).astype(int)
        probabilities = self.probabilities(eta=eta, thresholds=thresholds)
        cumulative = np.cumsum(probabilities, axis=-1)
        shape = np.broadcast_shapes(y.shape, cumulative.shape[:-1])
        y = np.broadcast_to(y, shape)
        cumulative = np.broadcast_to(cumulative, shape + (cumulative.shape[-1],))
        selected = np.take_along_axis(
            cumulative, np.clip(y, 0, cumulative.shape[-1] - 1)[..., None], axis=-1
        )[..., 0]
        return _return(np.where(y < 0, 0.0, np.where(y >= cumulative.shape[-1], 1.0, selected)))

    def rvs(self, size=None, *, rng=None, eta=0.0, thresholds=None, **_: Any):
        probabilities = self.probabilities(eta=eta, thresholds=thresholds)
        generator = _generator(rng)
        if probabilities.ndim != 1:
            raise ValueError("Ordinal rvs currently requires scalar eta")
        return generator.choice(len(probabilities), size=size, p=probabilities)

    def mean(self, *, eta=0.0, thresholds=None, **_: Any):
        probabilities = self.probabilities(eta=eta, thresholds=thresholds)
        return _return(np.sum(probabilities * np.arange(probabilities.shape[-1]), axis=-1))

    def variance(self, *, eta=0.0, thresholds=None, **_: Any):
        probabilities = self.probabilities(eta=eta, thresholds=thresholds)
        grid = np.arange(probabilities.shape[-1])
        mean = np.sum(probabilities * grid, axis=-1)
        return _return(np.sum(probabilities * (grid - mean[..., None]) ** 2, axis=-1))


class Multinomial(Family):
    """Categorical/multinomial-one-trial family.

    Supply either normalized ``probabilities`` or category ``logits``.  When
    logits has ``K-1`` columns, a zero baseline logit is appended.
    """

    name = "multinomial"
    support = "categories 0, ..., K-1"
    parameter_names = ("probabilities",)
    default_link = LOGIT
    discrete = True

    @staticmethod
    def _probabilities(probabilities=None, logits=None):
        if (probabilities is None) == (logits is None):
            raise ValueError("provide exactly one of probabilities= or logits=")
        if logits is not None:
            logits = _as_float(logits)
            logits = np.concatenate([logits, np.zeros(logits.shape[:-1] + (1,))], axis=-1)
            return special.softmax(logits, axis=-1)
        probabilities = _as_float(probabilities)
        if probabilities.ndim == 0 or probabilities.shape[-1] < 2:
            raise ValueError("multinomial probabilities need a final category dimension")
        if np.any(probabilities < 0) or np.any(~np.isfinite(probabilities)):
            raise ValueError("multinomial probabilities must be finite and nonnegative")
        total = probabilities.sum(axis=-1, keepdims=True)
        if np.any(total <= 0):
            raise ValueError("multinomial probability rows must have positive sum")
        return probabilities / total

    def log_prob(self, y, *, probabilities=None, logits=None, **_: Any):
        p = self._probabilities(probabilities, logits)
        y = _integer_observation(y).astype(int)
        shape = np.broadcast_shapes(y.shape, p.shape[:-1])
        y, p = np.broadcast_to(y, shape), np.broadcast_to(p, shape + (p.shape[-1],))
        valid = (y >= 0) & (y < p.shape[-1])
        selected = np.take_along_axis(p, np.clip(y, 0, p.shape[-1] - 1)[..., None], axis=-1)[..., 0]
        with np.errstate(divide="ignore"):
            return _return(np.where(valid, np.log(selected), -np.inf))

    def cdf(self, y, *, probabilities=None, logits=None, **_: Any):
        p = self._probabilities(probabilities, logits)
        y = np.floor(_as_float(y)).astype(int)
        shape = np.broadcast_shapes(y.shape, p.shape[:-1])
        y, p = np.broadcast_to(y, shape), np.broadcast_to(p, shape + (p.shape[-1],))
        cp = np.cumsum(p, axis=-1)
        selected = np.take_along_axis(cp, np.clip(y, 0, p.shape[-1] - 1)[..., None], axis=-1)[
            ..., 0
        ]
        return _return(np.where(y < 0, 0, np.where(y >= p.shape[-1], 1, selected)))

    def rvs(self, size=None, *, rng=None, probabilities=None, logits=None, **_: Any):
        p = self._probabilities(probabilities, logits)
        if p.ndim != 1:
            raise ValueError("Multinomial rvs currently requires one probability vector")
        return _generator(rng).choice(len(p), size=size, p=p)

    def mean(self, *, probabilities=None, logits=None, **_: Any):
        p = self._probabilities(probabilities, logits)
        return _return(np.sum(p * np.arange(p.shape[-1]), axis=-1))

    def variance(self, *, probabilities=None, logits=None, **_: Any):
        p = self._probabilities(probabilities, logits)
        grid = np.arange(p.shape[-1])
        mean = np.sum(p * grid, axis=-1)
        return _return(np.sum(p * (grid - mean[..., None]) ** 2, axis=-1))


class MixtureFamily(Family):
    """Base for composable mixture wrappers."""

    def __init__(self, conditional: Family, *, probability: Any = None, link=None, **predictors):
        if not isinstance(conditional, Family):
            raise TypeError("conditional must be a Family")
        if isinstance(probability, str):
            predictors["probability"] = probability
            probability = None
        super().__init__(link=link or LOGIT, **predictors)
        self.conditional = conditional
        self.fixed_probability = probability

    def _pi(self, probability):
        probability = self.fixed_probability if probability is None else probability
        return _probability(0.5 if probability is None else probability)


class ZeroInflated(MixtureFamily):
    """Point-mass-at-zero mixture with a normalized conditional family."""

    name = "zero-inflated"

    def log_prob(self, y, *, probability=None, **parameters):
        y = _as_float(y)
        pi = self._pi(probability)
        base = _as_float(self.conditional.log_prob(y, **parameters))
        zero_base = _as_float(self.conditional.log_prob(np.zeros_like(y), **parameters))
        at_zero = np.logaddexp(np.log(pi), np.log1p(-pi) + zero_base)
        return _return(np.where(y == 0, at_zero, np.log1p(-pi) + base))

    def cdf(self, y, *, probability=None, **parameters):
        y = _as_float(y)
        pi = self._pi(probability)
        base = _as_float(self.conditional.cdf(y, **parameters))
        return _return(np.where(y < 0, (1 - pi) * base, pi + (1 - pi) * base))

    def rvs(self, size=None, *, rng=None, probability=None, **parameters):
        generator = _generator(rng)
        structural = generator.random(size=size) < self._pi(probability)
        draws = self.conditional.rvs(size, rng=generator, **parameters)
        return np.where(structural, 0, draws)

    def mean(self, *, probability=None, **parameters):
        return _return((1 - self._pi(probability)) * _as_float(self.conditional.mean(**parameters)))

    def variance(self, *, probability=None, **parameters):
        pi = self._pi(probability)
        mean = _as_float(self.conditional.mean(**parameters))
        var = _as_float(self.conditional.variance(**parameters))
        return _return((1 - pi) * var + pi * (1 - pi) * mean**2)


class Hurdle(MixtureFamily):
    """Zero hurdle plus the conditional distribution truncated above zero."""

    name = "hurdle"

    def __init__(self, conditional: Family, **kwargs):
        if not conditional.discrete:
            raise NotImplementedError(
                "the reference Hurdle wrapper currently supports discrete families"
            )
        super().__init__(conditional, **kwargs)

    def _base_zero(self, parameters):
        return _as_float(self.conditional.cdf(0, **parameters))

    def log_prob(self, y, *, probability=None, **parameters):
        y = _as_float(y)
        pi = self._pi(probability)
        base_zero = self._base_zero(parameters)
        normalizer = np.log1p(-base_zero)
        positive = (
            np.log1p(-pi) + _as_float(self.conditional.log_prob(y, **parameters)) - normalizer
        )
        return _return(np.where(y == 0, np.log(pi), np.where(y > 0, positive, -np.inf)))

    def cdf(self, y, *, probability=None, **parameters):
        y = _as_float(y)
        pi = self._pi(probability)
        f0 = self._base_zero(parameters)
        f = _as_float(self.conditional.cdf(y, **parameters))
        positive = pi + (1 - pi) * (f - f0) / (1 - f0)
        return _return(np.where(y < 0, 0.0, positive))

    def rvs(self, size=None, *, rng=None, probability=None, **parameters):
        generator = _generator(rng)
        zero = generator.random(size=size) < self._pi(probability)
        result = np.asarray(self.conditional.rvs(size, rng=generator, **parameters))
        pending = (~zero) & (result <= 0)
        attempts = 0
        while np.any(pending):
            result[pending] = self.conditional.rvs(
                int(np.sum(pending)), rng=generator, **parameters
            )
            pending = (~zero) & (result <= 0)
            attempts += 1
            if attempts > 10_000:
                raise RuntimeError("hurdle rejection sampler failed to draw a positive value")
        return np.where(zero, 0, result)

    def mean(self, **parameters):
        raise NotImplementedError("generic hurdle moments require conditional positive moments")

    def variance(self, **parameters):
        raise NotImplementedError("generic hurdle moments require conditional positive moments")


class Truncated(Family):
    """Family conditioned on ``lower < Y <= upper``.

    For continuous families the distinction between open and closed endpoints
    is immaterial.  For discrete families ``lower`` is excluded and ``upper`` is
    included, which makes one-sided zero truncation ``lower=0`` explicit.
    """

    name = "truncated"

    def __init__(self, conditional: Family, *, lower=-np.inf, upper=np.inf):
        if not isinstance(conditional, Family):
            raise TypeError("conditional must be a Family")
        if np.any(_as_float(lower) >= _as_float(upper)):
            raise ValueError("truncation lower bound must be below upper bound")
        super().__init__(link=conditional.link)
        self.conditional, self.lower, self.upper = conditional, lower, upper
        self.discrete = conditional.discrete

    def _mass(self, parameters):
        lower_cdf = np.where(
            np.isneginf(self.lower), 0.0, _as_float(self.conditional.cdf(self.lower, **parameters))
        )
        upper_cdf = np.where(
            np.isposinf(self.upper), 1.0, _as_float(self.conditional.cdf(self.upper, **parameters))
        )
        mass = upper_cdf - lower_cdf
        if np.any(mass <= 0):
            raise ValueError("truncation interval has zero conditional probability")
        return mass

    def log_prob(self, y, **parameters):
        y = _as_float(y)
        inside = (y > _as_float(self.lower)) & (y <= _as_float(self.upper))
        return _return(
            np.where(
                inside,
                _as_float(self.conditional.log_prob(y, **parameters))
                - np.log(self._mass(parameters)),
                -np.inf,
            )
        )

    def cdf(self, y, **parameters):
        y = _as_float(y)
        lower_cdf = np.where(
            np.isneginf(self.lower), 0.0, _as_float(self.conditional.cdf(self.lower, **parameters))
        )
        raw = (_as_float(self.conditional.cdf(y, **parameters)) - lower_cdf) / self._mass(
            parameters
        )
        return _return(np.where(y <= self.lower, 0.0, np.where(y >= self.upper, 1.0, raw)))

    def rvs(self, size=None, *, rng=None, **parameters):
        generator = _generator(rng)
        result = np.asarray(self.conditional.rvs(size, rng=generator, **parameters))
        pending = (result <= self.lower) | (result > self.upper)
        attempts = 0
        while np.any(pending):
            result[pending] = self.conditional.rvs(
                int(np.sum(pending)), rng=generator, **parameters
            )
            pending = (result <= self.lower) | (result > self.upper)
            attempts += 1
            if attempts > 10_000:
                raise RuntimeError(
                    "truncated rejection sampler failed; interval probability is too small"
                )
        return result.item() if result.ndim == 0 else result


class Censored(Family):
    """Exact likelihood contributions for exact and censored observations.

    ``kind`` may be ``"exact"``, ``"left"``, ``"right"``, or ``"interval"``.
    Left censoring uses ``upper`` (or ``y``), right censoring uses ``lower`` (or
    ``y``), and interval censoring uses both endpoints.
    """

    name = "censored"

    def __init__(self, conditional: Family):
        if not isinstance(conditional, Family):
            raise TypeError("conditional must be a Family")
        super().__init__(link=conditional.link)
        self.conditional = conditional
        self.discrete = conditional.discrete

    def log_prob(self, y=None, *, kind="exact", lower=None, upper=None, **parameters):
        if kind == "exact":
            if y is None:
                raise ValueError("exact observations require y")
            return self.conditional.log_prob(y, **parameters)
        if kind == "left":
            endpoint = y if upper is None else upper
            if endpoint is None:
                raise ValueError("left censoring requires y or upper")
            return self.conditional.logcdf(endpoint, **parameters)
        if kind == "right":
            endpoint = y if lower is None else lower
            if endpoint is None:
                raise ValueError("right censoring requires y or lower")
            return self.conditional.logsf(endpoint, **parameters)
        if kind == "interval":
            if lower is None or upper is None:
                raise ValueError("interval censoring requires lower and upper")
            if np.any(_as_float(lower) >= _as_float(upper)):
                raise ValueError("interval lower endpoint must be below upper endpoint")
            return _return(
                _logdiffexp(
                    self.conditional.logcdf(upper, **parameters),
                    self.conditional.logcdf(lower, **parameters),
                )
            )
        raise ValueError("kind must be exact, left, right, or interval")

    def cdf(self, y, **parameters):
        return self.conditional.cdf(y, **parameters)

    def rvs(self, size=None, *, rng=None, **parameters):
        return self.conditional.rvs(size, rng=rng, **parameters)


class SurvivalFamily(Family):
    """Base class for right-censored time-to-event distributions."""

    support = "nonnegative time"

    def log_hazard(self, time: ArrayLike, **parameters):
        return _return(
            _as_float(self.log_prob(time, event=True, **parameters))
            - _as_float(self.logsf(time, **parameters))
        )

    def log_prob(self, time, *, event=True, **parameters):
        event = np.asarray(event, dtype=bool)
        log_density = _as_float(self.log_density(time, **parameters))
        log_survival = _as_float(self.logsf(time, **parameters))
        return _return(np.where(event, log_density, log_survival))

    def log_density(self, time: ArrayLike, **parameters):
        raise NotImplementedError


class Exponential(SurvivalFamily):
    """Exponential survival distribution with positive hazard ``rate``."""

    name = "exponential"
    parameter_names = ("rate",)
    default_link = LOG

    def log_density(self, time, *, rate=1.0, **_: Any):
        return _return(stats.expon.logpdf(_as_float(time), scale=1 / _positive(rate, "rate")))

    def cdf(self, time, *, rate=1.0, **_: Any):
        return _return(stats.expon.cdf(_as_float(time), scale=1 / _positive(rate, "rate")))

    def logsf(self, time, *, rate=1.0, **_: Any):
        return _return(stats.expon.logsf(_as_float(time), scale=1 / _positive(rate, "rate")))

    def rvs(self, size=None, *, rng=None, rate=1.0, **_: Any):
        return _generator(rng).exponential(1 / _positive(rate, "rate"), size=size)

    def mean(self, *, rate=1.0, **_: Any):
        return _return(1 / _positive(rate, "rate"))

    def variance(self, *, rate=1.0, **_: Any):
        return _return(1 / _positive(rate, "rate") ** 2)


class LogNormalSurvival(SurvivalFamily):
    """Lognormal accelerated-failure-time distribution.

    ``meanlog`` is the location of log time and ``sigma`` its standard
    deviation.  The event likelihood uses the density while a right-censored
    observation uses the survival probability.
    """

    name = "lognormal-survival"
    parameter_names = ("meanlog", "sigma")
    default_link = IDENTITY

    def log_density(self, time, *, meanlog=0.0, sigma=1.0, **_: Any):
        return _return(
            stats.lognorm.logpdf(
                _as_float(time), s=_positive(sigma, "sigma"), scale=np.exp(meanlog)
            )
        )

    def cdf(self, time, *, meanlog=0.0, sigma=1.0, **_: Any):
        return _return(
            stats.lognorm.cdf(_as_float(time), s=_positive(sigma, "sigma"), scale=np.exp(meanlog))
        )

    def logsf(self, time, *, meanlog=0.0, sigma=1.0, **_: Any):
        return _return(
            stats.lognorm.logsf(_as_float(time), s=_positive(sigma, "sigma"), scale=np.exp(meanlog))
        )

    def rvs(self, size=None, *, rng=None, meanlog=0.0, sigma=1.0, **_: Any):
        return _generator(rng).lognormal(meanlog, _positive(sigma, "sigma"), size=size)

    def mean(self, *, meanlog=0.0, sigma=1.0, **_: Any):
        return LogNormal().mean(meanlog=meanlog, sigma=sigma)

    def variance(self, *, meanlog=0.0, sigma=1.0, **_: Any):
        return LogNormal().variance(meanlog=meanlog, sigma=sigma)


class Weibull(SurvivalFamily):
    """Weibull survival distribution with shape and scale."""

    name = "weibull"
    parameter_names = ("shape", "scale")
    default_link = LOG

    def log_density(self, time, *, shape=1.0, scale=1.0, **_: Any):
        return _return(
            stats.weibull_min.logpdf(
                _as_float(time), _positive(shape, "shape"), scale=_positive(scale, "scale")
            )
        )

    def cdf(self, time, *, shape=1.0, scale=1.0, **_: Any):
        return _return(
            stats.weibull_min.cdf(
                _as_float(time), _positive(shape, "shape"), scale=_positive(scale, "scale")
            )
        )

    def logsf(self, time, *, shape=1.0, scale=1.0, **_: Any):
        return _return(
            stats.weibull_min.logsf(
                _as_float(time), _positive(shape, "shape"), scale=_positive(scale, "scale")
            )
        )

    def rvs(self, size=None, *, rng=None, shape=1.0, scale=1.0, **_: Any):
        return _positive(scale, "scale") * _generator(rng).weibull(
            _positive(shape, "shape"), size=size
        )

    def mean(self, *, shape=1.0, scale=1.0, **_: Any):
        shape, scale = _positive(shape, "shape"), _positive(scale, "scale")
        return _return(scale * special.gamma(1 + 1 / shape))

    def variance(self, *, shape=1.0, scale=1.0, **_: Any):
        shape, scale = _positive(shape, "shape"), _positive(scale, "scale")
        return _return(
            scale**2 * (special.gamma(1 + 2 / shape) - special.gamma(1 + 1 / shape) ** 2)
        )


class Gompertz(SurvivalFamily):
    """Gompertz distribution with hazard ``rate * exp(shape*time)``."""

    name = "gompertz"
    parameter_names = ("rate", "shape")
    default_link = LOG

    @staticmethod
    def _shape(value):
        result = _as_float(value)
        if np.any(~np.isfinite(result)) or np.any(result < 0):
            raise ValueError(
                "Gompertz shape must be nonnegative; negative shape gives a defective cure model"
            )
        return result

    def _cumhaz(self, time, rate, shape):
        time = _as_float(time)
        rate = _positive(rate, "rate")
        shape = self._shape(shape)
        return rate * time * special.exprel(shape * time)

    def log_density(self, time, *, rate=1.0, shape=0.1, **_: Any):
        time = _as_float(time)
        shape = self._shape(shape)
        return _return(
            np.where(
                time >= 0,
                np.log(_positive(rate, "rate")) + shape * time - self._cumhaz(time, rate, shape),
                -np.inf,
            )
        )

    def cdf(self, time, *, rate=1.0, shape=0.1, **_: Any):
        time = _as_float(time)
        return _return(np.where(time < 0, 0.0, -np.expm1(-self._cumhaz(time, rate, shape))))

    def logsf(self, time, *, rate=1.0, shape=0.1, **_: Any):
        time = _as_float(time)
        return _return(np.where(time < 0, 0.0, -self._cumhaz(time, rate, shape)))

    def rvs(self, size=None, *, rng=None, rate=1.0, shape=0.1, **_: Any):
        rate, shape = _positive(rate, "rate"), self._shape(shape)
        e = _generator(rng).exponential(size=size)
        denominator = np.where(np.abs(shape) < 1e-10, 1.0, shape)
        transformed = np.log1p(shape * e / rate) / denominator
        return np.where(np.abs(shape) < 1e-10, e / rate, transformed)

    def mean(self, *, rate=1.0, shape=0.1, **_: Any):
        rate, shape = _positive(rate, "rate"), self._shape(shape)
        rate, shape = np.broadcast_arrays(rate, shape)
        out = np.empty(rate.shape)
        zero = np.abs(shape) < 1e-10
        out[zero] = 1 / rate[zero]
        if np.any(~zero):
            out[~zero] = stats.gompertz.mean(rate[~zero] / shape[~zero], scale=1 / shape[~zero])
        return _return(out)

    def variance(self, *, rate=1.0, shape=0.1, **_: Any):
        rate, shape = _positive(rate, "rate"), self._shape(shape)
        rate, shape = np.broadcast_arrays(rate, shape)
        out = np.empty(rate.shape)
        zero = np.abs(shape) < 1e-10
        out[zero] = 1 / rate[zero] ** 2
        if np.any(~zero):
            out[~zero] = stats.gompertz.var(rate[~zero] / shape[~zero], scale=1 / shape[~zero])
        return _return(out)


class LogLogistic(SurvivalFamily):
    """Log-logistic survival distribution with positive shape and scale."""

    name = "log-logistic"
    parameter_names = ("shape", "scale")
    default_link = LOG

    def log_density(self, time, *, shape=1.0, scale=1.0, **_: Any):
        t, a, b = _as_float(time), _positive(shape, "shape"), _positive(scale, "scale")
        z = t / b
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.log(a) - np.log(b) + (a - 1) * np.log(z) - 2 * np.log1p(z**a)
        return _return(np.where(t > 0, out, -np.inf))

    def cdf(self, time, *, shape=1.0, scale=1.0, **_: Any):
        t, a, b = _as_float(time), _positive(shape, "shape"), _positive(scale, "scale")
        return _return(np.where(t <= 0, 0.0, special.expit(a * np.log(t / b))))

    def logsf(self, time, *, shape=1.0, scale=1.0, **_: Any):
        t, a, b = _as_float(time), _positive(shape, "shape"), _positive(scale, "scale")
        return _return(np.where(t <= 0, 0.0, -np.log1p((t / b) ** a)))

    def rvs(self, size=None, *, rng=None, shape=1.0, scale=1.0, **_: Any):
        u = _generator(rng).uniform(size=size)
        return _positive(scale, "scale") * (u / (1 - u)) ** (1 / _positive(shape, "shape"))

    def mean(self, *, shape=1.0, scale=1.0, **_: Any):
        shape, scale = _positive(shape, "shape"), _positive(scale, "scale")
        return _return(np.where(shape > 1, scale * (np.pi / shape) / np.sin(np.pi / shape), np.inf))

    def variance(self, *, shape=1.0, scale=1.0, **_: Any):
        shape, scale = _positive(shape, "shape"), _positive(scale, "scale")
        mean = np.where(shape > 1, scale * (np.pi / shape) / np.sin(np.pi / shape), np.inf)
        second = np.where(
            shape > 2,
            scale**2 * (2 * np.pi / shape) / np.sin(2 * np.pi / shape),
            np.inf,
        )
        return _return(np.where(shape > 2, second - mean**2, np.inf))


class PiecewiseExponential(SurvivalFamily):
    """Piecewise-constant hazard with explicit interval break points."""

    name = "piecewise-exponential"
    parameter_names = ("rates", "breaks")
    default_link = LOG

    @staticmethod
    def _validate(rates, breaks):
        rates, breaks = _positive(rates, "rates"), _as_float(breaks)
        if rates.ndim != 1 or breaks.ndim != 1 or len(rates) != len(breaks) + 1:
            raise ValueError("rates must have exactly one more entry than breaks")
        if np.any(np.diff(breaks) <= 0) or np.any(breaks <= 0):
            raise ValueError("piecewise-exponential breaks must be positive and increasing")
        return rates, breaks

    def _components(self, time, rates, breaks):
        rates, breaks = self._validate(rates, breaks)
        time = _as_float(time)
        edges = np.concatenate([[0.0], breaks])
        widths = np.diff(np.concatenate([[0.0], breaks, [np.inf]]))
        index = np.searchsorted(breaks, time, side="right")
        cum_before = np.concatenate([[0.0], np.cumsum(rates[:-1] * widths[:-1])])
        hazard = rates[np.clip(index, 0, len(rates) - 1)]
        cumulative = cum_before[np.clip(index, 0, len(rates) - 1)] + hazard * (
            time - edges[np.clip(index, 0, len(rates) - 1)]
        )
        return time, hazard, cumulative

    def log_density(self, time, *, rates, breaks, **_: Any):
        time, hazard, cumulative = self._components(time, rates, breaks)
        return _return(np.where(time >= 0, np.log(hazard) - cumulative, -np.inf))

    def cdf(self, time, *, rates, breaks, **_: Any):
        time, _, cumulative = self._components(time, rates, breaks)
        return _return(np.where(time < 0, 0.0, -np.expm1(-cumulative)))

    def logsf(self, time, *, rates, breaks, **_: Any):
        time, _, cumulative = self._components(time, rates, breaks)
        return _return(np.where(time < 0, 0.0, -cumulative))

    def rvs(self, size=None, *, rng=None, rates, breaks, **_: Any):
        rates, breaks = self._validate(rates, breaks)
        target = _generator(rng).exponential(size=size)
        result = np.zeros_like(target, dtype=float)
        remaining = np.asarray(target, dtype=float)
        left = 0.0
        active = np.ones_like(remaining, dtype=bool)
        for index, rate in enumerate(rates):
            width = np.inf if index == len(breaks) else breaks[index] - left
            capacity = rate * width
            finish = active & (remaining <= capacity)
            result[finish] = left + remaining[finish] / rate
            remaining = np.where(active & ~finish, remaining - capacity, remaining)
            active &= ~finish
            if index < len(breaks):
                left = breaks[index]
        return result.item() if result.ndim == 0 else result

    def mean(self, *, rates, breaks, **_: Any):
        rates, breaks = self._validate(rates, breaks)
        value = integrate.quad(
            lambda time: float(np.exp(self.logsf(time, rates=rates, breaks=breaks))),
            0,
            np.inf,
            epsabs=1e-10,
        )[0]
        return float(value)

    def variance(self, *, rates, breaks, **_: Any):
        rates, breaks = self._validate(rates, breaks)
        mean = self.mean(rates=rates, breaks=breaks)
        second = (
            2
            * integrate.quad(
                lambda time: time * float(np.exp(self.logsf(time, rates=rates, breaks=breaks))),
                0,
                np.inf,
                epsabs=1e-10,
            )[0]
        )
        return float(second - mean**2)


# Familiar aliases for common distribution names.
Normal = Gaussian
Student = StudentT
Lognormal = LogNormal
InverseGauss = InverseGaussian
NegativeBinomial = NegativeBinomial2
NB1 = NegativeBinomial1
NB2 = NegativeBinomial2
GenPoisson = GeneralizedPoisson
ConwayMaxwellPoisson = COMPoisson
ZeroInflatedFamily = ZeroInflated
HurdleFamily = Hurdle
TruncatedFamily = Truncated
CensoredFamily = Censored
ExponentialSurvival = Exponential
WeibullSurvival = Weibull
GompertzSurvival = Gompertz
LogLogisticSurvival = LogLogistic


__all__ = [
    "CAUCHIT",
    "CLOGLOG",
    "IDENTITY",
    "INVERSE",
    "INVERSE_SQUARED",
    "LOG",
    "LOGIT",
    "NB1",
    "NB2",
    "PROBIT",
    "Bernoulli",
    "Beta",
    "Binomial",
    "COMPoisson",
    "Censored",
    "CensoredFamily",
    "ConwayMaxwellPoisson",
    "Exponential",
    "ExponentialSurvival",
    "Family",
    "Gamma",
    "Gaussian",
    "GenPoisson",
    "GeneralizedPoisson",
    "Gompertz",
    "GompertzSurvival",
    "Hurdle",
    "HurdleFamily",
    "InverseGauss",
    "InverseGaussian",
    "Link",
    "LogLogistic",
    "LogLogisticSurvival",
    "LogNormal",
    "LogNormalSurvival",
    "Lognormal",
    "Multinomial",
    "NegativeBinomial",
    "NegativeBinomial1",
    "NegativeBinomial2",
    "Normal",
    "Ordinal",
    "PiecewiseExponential",
    "Poisson",
    "Student",
    "StudentT",
    "Truncated",
    "TruncatedFamily",
    "Tweedie",
    "Weibull",
    "WeibullSurvival",
    "ZeroInflated",
    "ZeroInflatedFamily",
    "get_link",
    "links",
]
