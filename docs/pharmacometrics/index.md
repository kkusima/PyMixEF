# Pharmacometrics

PyMixEF provides four connected layers: canonical event data, closed-form PK,
event-aware ODE simulation, and typed pharmacometric model declarations that
compile to the shared {py:class}`pymixef.ir.ModelIR`.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Event records and ODEs
:link: events-and-ode
:link-type: doc

Canonical dosing/observation semantics, deterministic same-time order,
ADDL/infusion expansion, provenance, numerical integration, and sensitivities.
:::

:::{grid-item-card} Closed-form PK and error
:link: pk-models
:link-type: doc

One- and two-compartment bolus, infusion, and oral profiles; residual-error and
censoring likelihood helpers.
:::

:::{grid-item-card} Typed model authoring
:link: authoring
:link-type: doc

Constrained parameters, ETAs, covariates, states, dose mappings, differential
equations, observation models, capability validation, and ModelIR compilation.
:::

:::{grid-item-card} Population-estimation primitives
:link: population-estimation
:link-type: doc

Random-effect transforms, conditional modes, Laplace population objectives,
shrinkage, finite differences, and the exact boundary of experimental SAEM and
unavailable FOCEI.
:::
::::

```{admonition} Calculation layers are intentionally distinct
:class: important

Closed-form and ODE functions simulate from supplied parameters. The DSL
declares and validates a model. Conditional-mode/Laplace routines expose
estimation primitives. None of those alone is an integrated production
population-PK estimator.
```

For an end-to-end learning sequence, use
[event semantics](../tutorials/07-pharmacometrics-event-semantics.md),
[PK and ODE simulation](../tutorials/08-closed-form-pk-and-ode.md), and
[DSL to ModelIR](../tutorials/09-pharmacometrics-dsl-and-model-ir.md).

```{toctree}
:maxdepth: 1

events-and-ode
pk-models
authoring
population-estimation
```
