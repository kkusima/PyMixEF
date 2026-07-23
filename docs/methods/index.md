# Statistical methods

Choose a method from the response distribution, dependence structure, and
scientific estimand. Each guide connects the model definition to the fitted
objects, diagnostics, and worked examples that implement it.

::::{grid} 1 3 3 3
:gutter: 2

:::{grid-item-card} Linear mixed models
:link: lmm
:link-type: doc

Continuous Gaussian outcomes with grouped random intercepts or slopes. Covers
ML and REML, covariance estimation, conditional and population prediction, and
simulation.
:::

:::{grid-item-card} Generalized mixed models
:link: glmm
:link-type: doc

Binary and count outcomes with Gaussian random effects. Covers supported
families, conditional interpretation, first-order Laplace fitting, calibration,
and simulation.
:::

:::{grid-item-card} Repeated-measures models
:link: mmrm
:link-type: doc

Continuous longitudinal outcomes with structured within-subject residual
covariance. Covers visit ordering, missing-response handling, contrasts,
degrees of freedom, and covariance diagnostics.
:::
::::

## Choose quickly

| Scientific structure | Start with | Primary dependence model |
|---|---|---|
| Continuous response; groups or nested units | [LMM](lmm.md) | Random effects |
| Binary or count response; grouped observations | [GLMM](glmm.md) | Gaussian random effects on the link scale |
| Continuous response at scheduled visits | [MMRM](mmrm.md) | Structured residual covariance |

If both a random trajectory and a repeated-measures residual structure appear
plausible, decide which representation matches the scientific question before
selecting an engine. The [analysis chooser](../getting-started/choose-analysis.md)
and [analysis matrix](../reference/analysis-matrix.md) compare the supported
execution paths in more detail.

```{toctree}
:maxdepth: 2

lmm
glmm
mmrm
```
