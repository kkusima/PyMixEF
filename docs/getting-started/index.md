# Getting started

This section takes you from an empty environment to an inspected, fitted, and
saved mixed-effects model. It also explains how to choose the analysis family
before writing code.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 1 · Install PyMixEF
:link: installation
:link-type: doc

Pick the smallest install that supports your workflow, verify the version, and
understand Python and optional-dependency requirements.
:::

:::{grid-item-card} 2 · Fit a first model
:link: quickstart
:link-type: doc

Run a complete LMM, inspect convergence, compare prediction modes, and save a
portable result.
:::

:::{grid-item-card} 3 · Choose an analysis
:link: choose-analysis
:link-type: doc

Use endpoint type, clustering, repeated-measures covariance, and scientific aim
to select LMM, GLMM, MMRM, PK/ODE, or the typed DSL.
:::

:::{grid-item-card} 4 · Learn the core workflow
:link: core-workflow
:link-type: doc

Understand the separation between model declaration, validation, compilation,
estimation, interpretation, diagnostics, and evidence preservation.
:::
::::

```{toctree}
:maxdepth: 1

installation
quickstart
choose-analysis
core-workflow
```

