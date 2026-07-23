# Mixed models for repeated measures

The MMRM path models a continuous longitudinal response with a fixed-effect mean
and a dense, structured residual covariance within each subject. Subjects are
independent blocks.

## When to use MMRM

MMRM is a useful starting point when:

- measurements occur on a finite, scientifically ordered visit axis;
- the target is a population mean or treatment contrast at one or more visits;
- within-subject residual covariance is central;
- likelihood analysis under a stated missing-at-random assumption is planned.

The current path does not combine structured MMRM residual covariance with
formula random effects. Use the LMM engine when random trajectories are the
chosen dependence representation.

## Declare covariance and mean

```python
import pymixef
from pymixef.covariance import AR1

residual = AR1(index="visit", group="subject")
model = pymixef.Model.from_formula(
    "change ~ baseline + treatment * visit",
    residual=residual,
)

plan = model.compile(
    data,
    engine="mmrm",
    method="reml",
    df_method="satterthwaite",
    maxiter=1000,
)
print(plan.explain())
result = plan.fit()
```

Supported structures include diagonal, unstructured, compound symmetry, AR(1),
heterogeneous AR(1), Toeplitz, heterogeneous Toeplitz, ante-dependence, spatial
power, and known covariance.

The residual declaration is an
{py:class}`pymixef.covariance.CovarianceStructure`. For example,
{py:class}`pymixef.covariance.AR1` defines

$$
\operatorname{Cov}(Y_{ij},Y_{ik}\mid X_i)
=\sigma^2\rho^{|t_j-t_k|},\qquad |\rho|<1,
$$

while {py:class}`pymixef.covariance.Unstructured` estimates every unique element
of the visit covariance subject to positive definiteness.

## Visit order is scientific state

Covariance adjacency is not derived from row order. Numeric visit labels and
numeric visit times are ordered ascending. Explicit visit order and ordered
categoricals are honored. Order-dependent structures reject ambiguous
nonnumeric labels.

Review:

```python
print(result.extra["visit_levels"])
print(result.extra["visit_times"])
print(result.extra["visit_order_source"])
print(result.extra["visit_covariance"])
```

Archiving the axis makes the covariance invariant to input-row permutations.

## Missing responses

Likelihood-based MMRM can use all observed outcomes under its modeled covariance
without filling missing responses. PyMixEF’s row audit preserves every
exclusion and reason.

Missing at random is an analysis assumption conditional on observed information;
optimizer convergence cannot verify it. Examine missingness patterns, treatment
and outcome associations, dropout reasons, and design-specific sensitivity
analyses.

## Fixed effects and contrasts

Treatment-by-visit terms allow the between-arm contrast to vary over visits.
Construct any contrast against the archived fixed-effect ordering and
covariance:

$$
\widehat{\Delta}_v=c_v^T\widehat{\beta},\qquad
\operatorname{SE}(\widehat{\Delta}_v)
=\sqrt{c_v^T\widehat{\operatorname{Var}}(\widehat{\beta})c_v}.
$$

{py:func}`pymixef.backends.mmrm.linear_inference` (aliases
{py:func}`pymixef.backends.mmrm.estimated_marginal_means` and
{py:func}`pymixef.backends.mmrm.contrasts`) retains
the coefficient mapping used for the calculation.

## Degrees of freedom

Supported output labels distinguish:

- residual degrees of freedom;
- Satterthwaite delta-method;
- a separately named KR-inspired approximation where available.

PyMixEF does not label KR-inspired output as exact Kenward–Roger. If Hessian
calculation is disabled, do not claim Hessian-dependent uncertainty evidence.

## Covariance checks

Inspect the fitted matrix, eigenvalues, correlations, and convergence boundaries.
A positive-definite estimate is necessary but does not establish that AR(1),
unstructured, or another form is scientifically correct.

An unstructured $m$-visit covariance uses $m(m+1)/2$ parameters. A flexible
model can be unstable with few subjects or sparse visit patterns. Prespecify
plausible alternatives and assess estimand sensitivity.

## Diagnostics

Useful views include:

- raw and fitted trajectories by arm;
- residual distributions and means by visit;
- fitted covariance/correlation heatmap;
- visit-specific counts and missingness;
- treatment contrasts with explicitly labeled uncertainty;
- covariance and mean-structure sensitivity.

Residual centering or an attractive heatmap is supportive evidence, not proof of
the missingness assumption or mean/covariance adequacy.

## Reference-engine scope

The implementation is a dense experimental reference calculation. It is not a
qualified confirmatory clinical-trial engine, and it does not claim exact
Kenward–Roger parity. A confirmatory workflow must prespecify estimand, target
visit/contrast, covariance strategy, DF method, missing-data assumptions,
multiplicity, and independent validation.

## Worked examples

- [Catalyst deactivation](../tutorials/03-catalyst-deactivation-mmrm.md): six
  cycles, AR(1), interaction-derived slopes, covariance heatmap.
- [Clinical-trial MMRM](../tutorials/05-clinical-trial-mmrm.md): four visits,
  four audited missing outcomes, adjusted treatment contrasts.

## API

- {py:mod}`pymixef.covariance`
- {py:mod}`pymixef.backends.mmrm`
- {py:mod}`pymixef.data`
