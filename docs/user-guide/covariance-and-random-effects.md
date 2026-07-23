# Covariance and random effects

PyMixEF represents between-group random effects and within-unit residual
covariance separately. The distinction determines the likelihood,
interpretation, and prediction modes.

## Formula random effects

```python
model = pymixef.Model.from_formula(
    "y ~ treatment * time + (1 + time | subject) + (1 | site)"
)
```

- `(1 | site)` creates a site random-intercept block.
- `(1 + time | subject)` creates a correlated subject intercept/slope block.
- `||` requests an independent/diagonal block.
- Multiple blocks are compiled separately with explicit grouping levels and
  covariance parameter slices.

Estimated random effects are conditional modes, subject to shrinkage. They are
not directly observed effects and should not be used as unqualified rankings of
subjects, centers, batches, or facilities.

## Residual covariance structures

```python
from pymixef.covariance import AR1

residual = AR1(index="visit", group="subject")
model = pymixef.Model.from_formula(
    "change ~ baseline + treatment * visit",
    residual=residual,
)
```

| Structure | Pattern | Common use |
|---|---|---|
| {py:class}`~pymixef.covariance.Diagonal` | independent, optionally heterogeneous variances | no residual correlation |
| {py:class}`~pymixef.covariance.Unstructured` | every variance/covariance estimated | few visits, flexible covariance |
| {py:class}`~pymixef.covariance.CompoundSymmetry` | common variance and correlation | exchangeable repeated values |
| {py:class}`~pymixef.covariance.AR1` | correlation decays geometrically with lag | equally ordered visits |
| {py:class}`~pymixef.covariance.HeterogeneousAR1` | AR(1) correlation, visit-specific SDs | ordered visits with changing variance |
| {py:class}`~pymixef.covariance.Toeplitz` | separate correlation by lag | lag-specific dependence |
| {py:class}`~pymixef.covariance.HeterogeneousToeplitz` | lag correlations, visit-specific SDs | flexible ordered covariance |
| {py:class}`~pymixef.covariance.AnteDependence` | sequential conditional dependence | ordered longitudinal measures |
| {py:class}`~pymixef.covariance.SpatialPower` | correlation by numeric distance | irregular numeric times/distances |
| {py:class}`~pymixef.covariance.KnownCovariance` | supplied positive-definite matrix | externally specified covariance |

Aliases ending in `Covariance` map to the corresponding structures.

## Visit order is model state

AR(1), Toeplitz, ante-dependence, and spatial structures need a meaningful axis.
Numeric visit labels/times are ordered ascending. An explicit `visit_order` or
ordered categorical axis is honored. Ambiguous nonnumeric ordering is refused
instead of inferred from input row order.

MMRM results archive:

- `visit_levels`;
- numeric `visit_times` where available;
- `visit_order_source`;
- the fitted `visit_covariance`.

This makes covariance adjacency reproducible after rows are shuffled or results
are reloaded.

## Parameterization and positive definiteness

Covariance parameters are optimized on unconstrained scales and mapped through
transform objects such as log/softplus and Cholesky covariance transforms.
Natural-scale SDs, correlations, and covariance matrices are reported.

Every covariance object supports a common contract:

- `parameter_count` and `parameter_names`;
- `covariance`/`matrix`;
- `validate`;
- numerical or analytic `derivatives` as implemented;
- `simulate`;
- `to_dict`.

Use {py:func}`pymixef.covariance.validate_covariance` for
symmetry/positive-definiteness checks and
{py:func}`pymixef.covariance.singularity_report` or
{py:func}`pymixef.diagnostics.covariance_singularity_table` to inspect small
eigenvalues and conditioning.

## Choosing complexity

An unstructured covariance is not automatically preferable: it consumes
$m(m+1)/2$ parameters for $m$ visits. Compare plausible structures based on
the design, number of units, visit coverage, stability, estimand sensitivity,
and prespecified selection rules. Positive definiteness is necessary but does
not establish a scientifically appropriate correlation model.

## Current engine boundaries

- Gaussian LMMs support formula random blocks and their covariance.
- MMRM models dense covariance within subjects and independence between
  subjects.
- The current MMRM path does not combine its residual covariance with formula
  random effects.
- The Laplace GLMM reference path supports compatible Gaussian random blocks,
  with conditional-mode convergence reported separately.

## API map

- {py:mod}`pymixef.covariance`
- {py:mod}`pymixef.transforms`
- {py:mod}`pymixef.random`
- {py:mod}`pymixef.diagnostics`
