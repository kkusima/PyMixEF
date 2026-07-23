# Inference and comparison

Inference must preserve the model, scale, covariance estimator, approximation,
degrees-of-freedom method, and nuisance-parameter treatment that generated it.

## Fixed-effect covariance

Fitted results retain `fixed_effect_names` and, when calculated,
`fixed_effect_covariance` in `result.extra`. This supports transparent Wald
standard errors and linear functions.

For a contrast vector $c$,

$$
\widehat{\Delta}=c^T\widehat{\beta}, \qquad
\operatorname{SE}(\widehat{\Delta})
=\sqrt{c^T\widehat{\operatorname{Var}}(\widehat{\beta})c}.
$$

Build contrasts against the archived coefficient ordering, not an assumed
column order.

## MMRM linear inference

{py:func}`pymixef.backends.mmrm.linear_inference` (aliases
{py:func}`pymixef.backends.mmrm.estimated_marginal_means` and
{py:func}`pymixef.backends.mmrm.contrasts`) evaluates named linear functions of
fixed effects with the selected, explicitly labeled degrees-of-freedom path.

Current labels distinguish:

- residual degrees of freedom;
- Satterthwaite delta-method;
- KR-inspired calculations where available.

KR-inspired output is not described as exact Kenward–Roger. A confirmatory
analysis should prespecify target visits, contrasts, multiplicity handling, and
DF method.

## Bootstrap

{py:func}`pymixef.inference.bootstrap` performs deterministic row or cluster resampling through a
caller-supplied fit function.

```python
from pymixef import bootstrap, fit

def refit(sample):
    return fit("y ~ x + (1 | subject)", sample, method="reml")

boot = bootstrap(
    refit,
    data,
    n_replicates=500,
    seed=2026,
    cluster="subject",
    checkpoint="bootstrap.json",
    resume=True,
)
print(boot.intervals(level=0.95).to_dict())
```

The result retains successful and failed replicate accounting plus percentile
intervals. Cluster resampling should match the independent sampling unit;
resampling rows inside a clustered design generally answers a different
question.

## Compare a result to a reference

{py:func}`pymixef.compare.compare` creates a structured
{py:class}`pymixef.compare.ComparisonResult`:

```python
comparison = pymixef.compare(
    result,
    reference=reference_payload,
    mapping={"treatment[treated]": "treatment"},
    conventions={"likelihood": "normalized"},
)
print(comparison.to_dict())
```

Parameter mapping and likelihood conventions are explicit. Likelihood/AIC
comparisons are refused when conventions are incompatible.

## ML and REML comparisons

- Use ML rather than REML to compare models with different fixed-effect spaces
  under the usual nested-model reasoning.
- REML objectives from different fixed-effect designs do not share the same
  integrated fixed-effect constant and should not be compared as if they did.
- Covariance-structure comparisons and boundary cases require design-specific
  care; a naive chi-square reference may be inappropriate.

## Effect scale

- Gaussian identity-link coefficients are on the response scale.
- Logit coefficients are conditional log odds; exponentiation gives
  conditional odds ratios.
- Log-link coefficients become conditional multiplicative ratios after
  exponentiation.
- Transforming an interval endpoint-by-endpoint is appropriate for monotone
  transforms but does not turn a conditional estimand into a marginal one.

## API map

- {py:mod}`pymixef.inference`
- {py:mod}`pymixef.compare`
- {py:mod}`pymixef.backends.mmrm`
- [Likelihood and uncertainty conventions](../concepts/conventions.md)
