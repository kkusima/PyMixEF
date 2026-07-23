# Diagnostics and simulation

Diagnostics in PyMixEF are data-first: methods return machine-readable tables,
and plots are a view of those retained values.

## Residual diagnostics

```python
diagnostics = result.residual_diagnostics()
print(diagnostics.name)
print(diagnostics.columns)
print(len(diagnostics))
```

The standard residual table can contain source-aligned row IDs, observed and
fitted values, raw residuals, and Pearson residuals. Pass `observed` or
`variance` explicitly when the archived fit does not contain the inputs needed
for a desired residual definition.

Useful plots include:

- residuals versus fitted values;
- residuals versus time/visit or key predictors;
- distributions or Q–Q views, interpreted in context;
- group-level summaries;
- variance/correlation heatmaps;
- random-effect conditional modes with uncertainty where available.

No single plot certifies adequacy. Look for patterns tied to the model’s mean,
variance, dependence, link, influential groups, and design.

## Named diagnostic tables

`result.diagnostic(name)` retrieves an archived diagnostic by name, such as
random effects where supported. `pymixef.diagnostics` also provides:

- `residual_table`;
- `vpc_table`;
- `covariance_singularity_table`;
- the immutable `DiagnosticTable` container.

## Reproducible simulation

```python
simulated = result.simulate(
    n_replicates=1000,
    seed=2026,
    parameter_uncertainty="none",
    random_effects=True,
    residual_error=True,
    output="numpy",
)
```

The seed controls the package’s named random streams. Record whether random
effects, residual error, and parameter uncertainty were included. Simulations
use the archived fitted covariance and model conventions; they are not external
validation data.

## Visual predictive checks

```python
vpc = result.vpc(
    simulations=1000,
    seed=2026,
    bins="adaptive",
    prediction_corrected=False,
)
```

The VPC table contains bin definitions, observed quantiles, simulated medians
and envelopes, and observed counts. Explicit numeric bin edges are useful when
scientific time intervals matter.

A trajectory inside an envelope is one calibration view. It is not a universal
acceptance criterion, and sparse bins can hide local misspecification.

## Random streams

`RandomStreamManager` and `random_streams(seed)` create deterministic named
substreams so adding one stochastic component need not silently perturb all
others. Preserve the top-level seed, stream names, package version, and model
hash.

## Prediction modes

```python
population = result.prediction(mode="population")
conditional = result.prediction(mode="conditional")
```

Population predictions use the fixed-effect/typical-group mean. Conditional
predictions include fitted random effects. New-group prediction requires
integrating or simulating new random effects as explicitly requested by the
supporting path. Do not compare diagnostics across modes without labeling the
change.

## A practical diagnostic sequence

1. Verify trustworthy convergence.
2. Reconcile observations to source rows.
3. Plot observed and fitted values on meaningful axes.
4. Inspect residual location, scale, dependence, and group patterns.
5. Inspect covariance conditioning/boundaries and random-effect shrinkage.
6. Run seeded predictive simulation and VPC summaries.
7. Repeat prespecified covariance, mean-structure, missingness, and
   approximation sensitivities.
8. Preserve tables, plots, seeds, and conclusions together.

The ten [worked tutorials](../tutorials/index.md) show this sequence with
validated plots and interpretation immediately next to each figure.

