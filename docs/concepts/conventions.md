# Likelihood, prediction, and uncertainty conventions

PyMixEF log likelihoods include data-dependent normalization constants unless a
method document explicitly says that only an unnormalized research objective is
available. AIC or likelihood comparison is refused when conventions differ.

Gaussian maximum likelihood uses

$$
\ell(\beta,\theta)=-\frac12\{n\log(2\pi)+\log|V_\theta|
 +(y-X\beta)^\mathsf{T} V_\theta^{-1}(y-X\beta)\}.
$$

REML additionally integrates the fixed effects and records the exact constant
convention in the result. Approximate likelihoods name the approximation
(`laplace`, for example) in the result and report sensitivity controls.

Prediction modes are never implicit:

- `population`: fixed-effects/typical-subject prediction;
- `conditional`: includes empirical conditional random effects;
- `new-subject`: integrates or simulates new random effects as requested.

Every standard error identifies the covariance estimator, calculation scale,
approximation, degrees-of-freedom method, and nuisance-parameter treatment in
result metadata.

## Objective comparison

{py:func}`pymixef.compare.compare` calculates objective differences only when
the PyMixEF result
and reference both declare:

- estimation method;
- objective convention;
- whether likelihood normalization constants are included.

Parameter estimates can still be aligned when objective conventions differ, but
the objective difference remains unavailable. This prevents an apparently
precise comparison of non-equivalent numbers.

## ML, REML, and approximate likelihoods

- Gaussian ML objectives are comparable across compatible fixed-effect models
  under the same data and convention.
- REML integrates/adjusts for the fixed-effect space; REML objectives from
  different fixed-effect designs are not ordinary likelihood comparisons.
- Laplace GLMM objectives are conditional on the named approximation and
  controls. Report “first-order Laplace” rather than only “maximum likelihood.”
- Boundary variance tests can have nonstandard reference distributions; a naive
  chi-square LRT is not universal.

## Conditional, population, marginal, and new-group

These terms are not interchangeable.

| Output | Random-effect treatment | Answers |
|---|---|---|
| conditional | fitted modes for observed groups | fitted response for these groups |
| population/typical-group | random effect set to its reference value | fixed-effect curve at $b=0$ |
| marginal | integrate over the random-effect distribution | population-average response under the model |
| new-group simulation | draw a new random effect | predictive distribution for an unseen group |

With an identity link, the typical-group and random-effect-marginal mean can
coincide. With nonlinear links, they generally do not.

For a GLMM with inverse link $g^{-1}$, the distinction is explicit:

$$
\mu_\mathrm{typical}(x)=g^{-1}(x^\mathsf{T}\beta),
\qquad
\mu_\mathrm{marginal}(x)
=\int g^{-1}(x^\mathsf{T}\beta+z^\mathsf{T}b)\,
\phi(b;0,G)\,db.
$$

Except for special link/distribution combinations, applying the inverse link
before integrating is not the same as applying it after setting $b=0$.

## Link and response scales

Coefficients are estimated on the linear-predictor/link scale:

- identity: response-unit difference;
- logit: log odds; exponentiation gives a conditional odds ratio;
- log: log mean/rate; exponentiation gives a conditional ratio;
- other links require their own inverse transformation.

An odds ratio is not a risk ratio, probability ratio, or percentage-point
change. Always state the link, conditional/marginal target, covariate reference
values, and random-effect treatment.

## Standard errors and intervals

Before reporting an interval, identify:

1. parameter/contrast and calculation scale;
2. covariance source (for example observed Hessian);
3. approximation and whether nuisance uncertainty is included;
4. reference distribution and degrees of freedom;
5. transformation to the reported scale;
6. multiplicity and prespecification where relevant.

If Hessian diagnostics are indefinite or unavailable, ordinary Hessian-based
Wald intervals are not supported by the calculation.

## Simulation uncertainty

Seeded simulation is reproducible, but finite replicates carry Monte Carlo
error. Increase replicates until the simulation summary is stable for the
decision scale and report the seed, replicate count, random-effect/residual
components, and any parameter uncertainty.

## Missing data

A likelihood calculated under missing at random does not verify that assumption.
The data audit establishes which records were used; scientific missingness
reasoning and sensitivity analysis establish how conclusions depend on
unobserved outcomes.
