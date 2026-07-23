from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from pymixef import families


def test_links_round_trip() -> None:
    probabilities = np.array([0.1, 0.4, 0.8])
    for link in (
        families.links.logit,
        families.links.probit,
        families.links.cloglog,
        families.links.cauchit,
    ):
        assert np.allclose(link.inverse(link(probabilities)), probabilities)
    positive = np.array([0.2, 1.0, 3.0])
    assert np.allclose(families.links.log.inverse(families.links.log(positive)), positive)


@pytest.mark.parametrize(
    ("family", "y", "parameters", "reference"),
    [
        (
            families.Gaussian(),
            np.array([-1.0, 0.5]),
            {"mu": 0.2, "sigma": 1.3},
            stats.norm.logpdf(np.array([-1.0, 0.5]), 0.2, 1.3),
        ),
        (
            families.StudentT(df=5),
            np.array([-1.0, 0.5]),
            {"mu": 0.2, "sigma": 1.3},
            stats.t.logpdf(np.array([-1.0, 0.5]), 5, 0.2, 1.3),
        ),
        (
            families.Poisson(),
            np.arange(3),
            {"mu": 1.7},
            stats.poisson.logpmf(np.arange(3), 1.7),
        ),
        (
            families.Binomial(trials=5),
            np.arange(3),
            {"mu": 0.3},
            stats.binom.logpmf(np.arange(3), 5, 0.3),
        ),
    ],
)
def test_normalized_log_prob_matches_scipy(
    family: families.Family,
    y: np.ndarray,
    parameters: dict[str, float],
    reference: np.ndarray,
) -> None:
    assert family.normalized
    assert np.allclose(family.log_prob(y, **parameters), reference)


def test_continuous_parameterizations_and_moments() -> None:
    gamma = families.Gamma()
    inverse_gaussian = families.InverseGaussian()
    beta = families.Beta()
    assert gamma.mean(mu=3, dispersion=0.4) == 3
    assert gamma.variance(mu=3, dispersion=0.4) == pytest.approx(3.6)
    assert inverse_gaussian.variance(mu=2, dispersion=0.25) == pytest.approx(2)
    assert beta.variance(mu=0.4, precision=9) == pytest.approx(0.024)
    assert np.isfinite(gamma.log_prob(1.5, mu=3, dispersion=0.4))
    assert np.isfinite(inverse_gaussian.log_prob(1.5, mu=2, dispersion=0.25))
    assert np.isfinite(beta.log_prob(0.4, mu=0.4, precision=9))


def test_count_families_are_normalized_and_parameterized_explicitly() -> None:
    grid = np.arange(500)
    nb1 = families.NegativeBinomial1()
    nb2 = families.NegativeBinomial2()
    assert np.exp(nb1.log_prob(grid, mu=4, dispersion=0.7)).sum() == pytest.approx(1)
    assert np.exp(nb2.log_prob(grid, mu=4, dispersion=2.5)).sum() == pytest.approx(1)
    assert nb1.variance(mu=4, dispersion=0.7) == pytest.approx(6.8)
    assert nb2.variance(mu=4, dispersion=2.5) == pytest.approx(10.4)

    generalized = families.GeneralizedPoisson()
    assert np.exp(generalized.log_prob(grid, mu=3, dispersion=0.2)).sum() == pytest.approx(
        1, abs=1e-10
    )
    comp = families.COMPoisson()
    assert np.exp(comp.log_prob(grid, rate=2, dispersion=1.4)).sum() == pytest.approx(1, abs=1e-11)


def test_ordinal_and_multinomial_probabilities() -> None:
    ordinal = families.Ordinal(thresholds=[-1, 0.5])
    probabilities = ordinal.probabilities(eta=0.2)
    assert probabilities.sum() == pytest.approx(1)
    assert np.exp(ordinal.log_prob(np.arange(3), eta=0.2)).sum() == pytest.approx(1)

    multinomial = families.Multinomial()
    probabilities = np.array([0.2, 0.3, 0.5])
    assert np.exp(
        multinomial.log_prob(np.arange(3), probabilities=probabilities)
    ).sum() == pytest.approx(1)
    assert np.allclose(
        multinomial.cdf(np.arange(3), probabilities=probabilities), np.cumsum(probabilities)
    )


def test_mixture_truncation_and_censoring_are_normalized() -> None:
    poisson = families.Poisson()
    zero_inflated = families.ZeroInflated(poisson, probability=0.3)
    hurdle = families.Hurdle(poisson, probability=0.25)
    grid = np.arange(100)
    assert np.exp(zero_inflated.log_prob(grid, mu=2)).sum() == pytest.approx(1)
    assert np.exp(hurdle.log_prob(grid, mu=2)).sum() == pytest.approx(1)

    zero_truncated = families.Truncated(poisson, lower=0)
    assert np.exp(zero_truncated.log_prob(grid, mu=2)).sum() == pytest.approx(1)
    assert np.isneginf(zero_truncated.log_prob(0, mu=2))

    censored = families.Censored(families.Gaussian())
    expected = np.log(stats.norm.cdf(1) - stats.norm.cdf(-1))
    assert censored.log_prob(kind="interval", lower=-1, upper=1) == pytest.approx(expected)
    assert censored.log_prob(0, kind="left") == pytest.approx(np.log(0.5))


def test_tweedie_exact_scope_and_controlled_rng() -> None:
    tweedie = families.Tweedie(power=1.5)
    assert np.isfinite(tweedie.log_prob(0, mu=2, dispersion=0.5))
    assert np.isfinite(tweedie.log_prob(1, mu=2, dispersion=0.5))
    first = tweedie.rvs(20, rng=42, mu=2, dispersion=0.5)
    second = tweedie.rvs(20, rng=42, mu=2, dispersion=0.5)
    assert np.array_equal(first, second)
    with pytest.raises(NotImplementedError, match="normalized Tweedie"):
        tweedie.log_prob(1, mu=2, dispersion=0.5, power=2.5)
    with pytest.raises(NotImplementedError, match="CDF"):
        tweedie.cdf(1, mu=2, dispersion=0.5)


def test_survival_likelihood_uses_density_or_survival() -> None:
    exponential = families.Exponential()
    values = exponential.log_prob([1, 2], event=[True, False], rate=0.5)
    assert np.allclose(values, [np.log(0.5) - 0.5, -1.0])
    weibull = families.Weibull()
    assert np.isfinite(weibull.log_prob(1.2, event=True, shape=1.5, scale=2))
    piecewise = families.PiecewiseExponential()
    assert np.isfinite(piecewise.log_prob(2.5, event=False, rates=[0.2, 0.4], breaks=[2]))


def test_distribution_aliases() -> None:
    assert families.Normal is families.Gaussian
    assert families.NB2 is families.NegativeBinomial2
    assert isinstance(
        families.ZeroInflated(
            conditional=families.NegativeBinomial2(dispersion="~ treatment"),
            probability="~ baseline",
        ),
        families.ZeroInflated,
    )
