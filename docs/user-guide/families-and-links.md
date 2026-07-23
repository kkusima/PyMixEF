# Families and links

Family objects provide probability calculations, simulation, moments, and link
metadata. They are useful both inside compatible model engines and as standalone
distribution objects.

## Shared family contract

Depending on the distribution, a {py:class}`pymixef.families.Family` exposes
`log_prob`/`log_probability`,
`logpdf`, `logpmf`, `cdf`, `logcdf`, `sf`, `logsf`, `rvs`, `random`, `mean`,
`variance`, and `moments`. Methods validate support and parameter domains.

```python
import numpy as np
from pymixef.families import Bernoulli, Gaussian

normal = Gaussian()
log_density = normal.log_prob(np.array([0.0, 1.0]), mean=0.0, scale=1.0)

binary = Bernoulli()
draws = binary.rvs(mean=np.array([0.2, 0.8]), random_state=2026)
```

Consult the signature-level API because parameters differ by family.

## Link functions

Built-ins are `identity`, `log`, `logit`, `probit`, `cloglog`, `cauchit`,
`inverse`, and `inverse_squared`. Each {py:class}`pymixef.families.Link` defines
forward, inverse, and
derivative calculations with domain checks.

```python
from pymixef.families import get_link

logit = get_link("logit")
probability = logit.inverse(0.75)
```

The `links` namespace and uppercase constants such as `LOGIT` are convenience
forms.

## Distribution catalog

| Area | Families |
|---|---|
| Continuous | {py:class}`~pymixef.families.Gaussian`, {py:class}`~pymixef.families.StudentT`, {py:class}`~pymixef.families.LogNormal`, {py:class}`~pymixef.families.Gamma`, {py:class}`~pymixef.families.InverseGaussian`, {py:class}`~pymixef.families.Beta`, {py:class}`~pymixef.families.Tweedie` |
| Discrete | {py:class}`~pymixef.families.Bernoulli`, {py:class}`~pymixef.families.Binomial`, {py:class}`~pymixef.families.Poisson`, {py:class}`~pymixef.families.NegativeBinomial1`, {py:class}`~pymixef.families.NegativeBinomial2`, {py:class}`~pymixef.families.GeneralizedPoisson`, {py:class}`~pymixef.families.COMPoisson`, {py:class}`~pymixef.families.Ordinal`, {py:class}`~pymixef.families.Multinomial` |
| Composition | {py:class}`~pymixef.families.ZeroInflated`, {py:class}`~pymixef.families.Hurdle`, {py:class}`~pymixef.families.Truncated`, {py:class}`~pymixef.families.Censored` |
| Survival | {py:class}`~pymixef.families.Exponential`, {py:class}`~pymixef.families.LogNormalSurvival`, {py:class}`~pymixef.families.Weibull`, {py:class}`~pymixef.families.Gompertz`, {py:class}`~pymixef.families.LogLogistic`, {py:class}`~pymixef.families.PiecewiseExponential` |

Aliases such as `Normal`, `NB1`, `NB2`, `NegativeBinomial`, and
`WeibullSurvival` are indexed in the [alias reference](../api/aliases.md).

## Catalog support versus engine support

The broad catalog is a probability and representation layer. Current formula
fit support is narrower:

| Engine | Accepted family path |
|---|---|
| LMM | Gaussian with identity link |
| GLMM | Bernoulli, binomial, Poisson, and negative-binomial-2 with supported canonical-link parameterizations |
| MMRM | Gaussian continuous response with structured residual covariance |

Wrappers, survival families, ordinal/multinomial families, and other catalog
objects are not automatically executable by the current GLMM backend. Model
validation checks the requested engine/family/link combination before
optimization.

## Conditional mean and link scale

In a GLMM,

$$
g\{\operatorname{E}(Y_{ij}\mid b_i)\}=x_{ij}^{T}\beta+z_{ij}^{T}b_i.
$$

Coefficients live on the link scale. For a logit link, exponentiating a
coefficient produces a conditional odds ratio—not a probability ratio or
percentage-point difference. Setting $b_i=0$ gives a typical-cluster
conditional curve, not a random-effect-integrated marginal curve.

## Composition objects

{py:class}`pymixef.families.ZeroInflated`,
{py:class}`pymixef.families.Hurdle`,
{py:class}`pymixef.families.Truncated`, and
{py:class}`pymixef.families.Censored` wrap a base family while
retaining the component definition. They support relevant standalone
probability calculations and IR representation. Always confirm backend
compatibility before attempting a fit.

## API map

The complete signatures, parameters, aliases, and inherited methods are in
{py:mod}`pymixef.families`.
