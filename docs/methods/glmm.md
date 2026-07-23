# Generalized linear mixed models

A GLMM combines a non-Gaussian response family, link function, fixed effects,
and Gaussian random effects:

$$
g\{\operatorname{E}(Y_{ij}\mid b_i)\}
=x_{ij}^{T}\beta+z_{ij}^{T}b_i,\qquad b_i\sim N(0,G).
$$

PyMixEF’s current reference engine integrates supported random effects with a
first-order Laplace approximation around the joint conditional mode.

## Supported fit subset

| Family | Typical endpoint | Link/parameterization |
|---|---|---|
| {py:class}`~pymixef.families.Bernoulli` | one binary outcome per row | supported canonical logit path |
| {py:class}`~pymixef.families.Binomial` | successes out of trials | validated supported binomial path |
| {py:class}`~pymixef.families.Poisson` | count | supported canonical log path |
| {py:class}`~pymixef.families.NegativeBinomial2` | overdispersed count | supported NB2 mean/dispersion path |

The probability family catalog is broader. Catalog objects such as beta,
ordinal, zero-inflated, hurdle, survival, NB1, and COM-Poisson can support
standalone calculations or model representation without being accepted by this
engine.

## Fit a Bernoulli random-intercept model

```python
import pymixef

model = pymixef.Model.from_formula(
    "response ~ treatment + baseline + (1 | clinic)",
    family=pymixef.families.Bernoulli(),
)

validation = model.validate(data, engine="glmm", method="laplace")
validation.raise_for_errors()

plan = model.compile(
    data,
    engine="glmm",
    method="laplace",
    maxiter=400,
)
print(plan.explain())
result = plan.fit()
```

{py:meth}`pymixef.model.Model.validate` returns a structured
{py:class}`pymixef.model.ValidationReport`; its
{py:meth}`~pymixef.model.ValidationReport.raise_for_errors` method prevents an
invalid declaration from reaching {py:meth}`pymixef.model.Model.compile`.

## Laplace calculation

For population parameters $\theta$, the engine finds the random-effect
conditional mode and uses local curvature to approximate the random-effect
integral. The result retains:

- approximation name: `first-order Laplace at the joint conditional mode`;
- quadrature order `1` as an explicit calculation label;
- normalization/objective convention;
- inner conditional-mode failure count;
- outer optimizer state, gradient, boundaries, and warnings.

Adaptive Gauss–Hermite quadrature (AGHQ) is unavailable. Requesting `aghq`
raises a stable compatibility error; Laplace is not silently substituted.

Writing the joint log density as
$h_i(b_i;\theta)=\log p(y_i\mid b_i,\theta)+\log p(b_i\mid\theta)$, the
first-order Laplace contribution is

$$
\log p(y_i\mid\theta)
\approx h_i(\widehat b_i;\theta)
+\frac{q_i}{2}\log(2\pi)
-\frac12\log\left|-\nabla^2_{b_i}h_i(\widehat b_i;\theta)\right|.
$$

The reported approximation metadata on
{py:attr}`FitResult.extra <pymixef.results.FitResult.extra>` makes this
first-order calculation auditable.

## Interpret coefficients on the correct scale

For a logit model, a coefficient is a conditional log odds ratio:

```python
import numpy as np

conditional_odds_ratio = np.exp(result.parameters["treatment"])
```

It is not a risk ratio, probability ratio, or percentage-point change. A
probability curve with the random effect set to zero describes a typical
cluster conditional on $b=0$, not a population-marginal curve integrated over
the random-effect distribution.

For a log link, exponentiated coefficients are conditional multiplicative
effects on the mean.

## Random-effect modes

Conditional modes are shrunk estimates. Extreme observed cluster rates can
produce moderated modes because the model pools information. Modes should not
be treated as permanent clinic, lot, or facility performance scores; their
uncertainty, exposure, case mix, and estimation on the same outcomes matter.

## Diagnostics

Review both optimization levels:

- outer parameter optimization;
- cluster-level conditional-mode convergence;
- scaled gradient and warning codes;
- covariance boundary/singularity;
- fitted probabilities/means versus observed summaries;
- residual diagnostics appropriate to the endpoint;
- cluster-level influence and held-out-cluster performance;
- approximation sensitivity for consequential conclusions.

Binned calibration computed on the same data and fitted conditional modes is an
in-sample diagnostic. It is not external predictive validation.

## Simulation and predictions

{py:meth}`FitResult.simulate <pymixef.results.FitResult.simulate>` can generate
seeded replicate data under archived fixed and
random-effect covariance. Preserve whether new random effects and residual
sampling were included. Simulation checks consistency with the fitted model; it
does not prove that the data-generating assumptions are true.

## Reference-engine scope

The engine is a dense experimental first-order Laplace implementation. It does
not claim AGHQ order sensitivity, sparse production scaling, zero-inflation or
hurdle fitting, noncanonical-link generality, or a full glmmTMB-class feature
surface.

## Worked examples

- [Catalyst activation success](../tutorials/02-binary-catalyst-success-glmm.md)
- [Clustered medical response](../tutorials/06-binary-response-glmm.md)

Both show conditional probability curves, calibration, random modes, exact
executed results, and explicit interpretation boundaries.

## API

- {py:mod}`pymixef.families`
- {py:mod}`pymixef.backends.glmm`
- {py:mod}`pymixef.model`
