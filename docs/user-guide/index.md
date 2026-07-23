# User guide

The user guide follows the lifecycle of a statistical analysis. Method-specific
likelihood details live in the LMM, GLMM, and MMRM guides; every Python symbol is
also indexed in the [API reference](../api/index.md).

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Data and formulas
:link: data-and-formulas
:link-type: doc

Supported inputs, missingness contracts, source-row audits, formula grammar,
factor handling, and design-matrix inspection.
:::

:::{grid-item-card} Families and links
:link: families-and-links
:link-type: doc

Probability calculations, default links, wrappers, censoring/survival objects,
and the boundary between the family catalog and fit support.
:::

:::{grid-item-card} Covariance and random effects
:link: covariance-and-random-effects
:link-type: doc

Random blocks, structured residual covariance, visit ordering, parameterization,
validation, singularity checks, and simulation.
:::

:::{grid-item-card} Fitting and convergence
:link: fitting-and-convergence
:link-type: doc

Engine and method choice, compilation, numerical controls, structured
convergence, warning codes, and recovery workflow.
:::

:::{grid-item-card} Inference and comparison
:link: inference-and-comparison
:link-type: doc

Coefficient covariance, linear inference, bootstrap, result comparison, scale,
and degrees-of-freedom labels.
:::

:::{grid-item-card} Diagnostics and simulation
:link: diagnostics-simulation
:link-type: doc

Auditable residual tables, random effects, singularity, reproducible simulation,
and visual predictive checks.
:::

:::{grid-item-card} Results and provenance
:link: results-provenance
:link-type: doc

`FitResult`, manifests, hashes, portable archives, reports, traceability, and
validation bundles.
:::
::::

```{toctree}
:maxdepth: 1

data-and-formulas
families-and-links
covariance-and-random-effects
fitting-and-convergence
inference-and-comparison
diagnostics-simulation
results-provenance
```

