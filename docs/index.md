---
hide-toc: true
---

<div class="pymixef-hero">
  <p class="pymixef-kicker">PyMixEF 0.1 documentation</p>
  <h1>Mixed-effects modeling in Python</h1>
  <p>
    Specify, fit, diagnose, and preserve mixed-effects analyses with an
    inspectable model representation. Follow a statistical workflow, reproduce
    complete scientific examples, or look up the public Python API.
  </p>
  <div class="pymixef-actions">
    <a href="getting-started/quickstart/">Get started</a>
    <a href="api/">Browse the API reference</a>
  </div>
</div>

# Start with your task

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Install and start
:link: getting-started/index
:link-type: doc

Choose an installation profile, fit a first model, and learn the
declare → validate → compile → fit → diagnose workflow.
:::

:::{grid-item-card} Choose an analysis
:link: getting-started/choose-analysis
:link-type: doc

Map a scientific question to LMM, GLMM, MMRM, PK/ODE simulation, or a
pharmacometric model declaration.
:::

:::{grid-item-card} Work through examples
:link: tutorials/index
:link-type: doc

Ten pre-executed, assertion-backed case studies: three materials/catalysis
analyses and seven biomedical/pharmaceutical workflows.
:::

:::{grid-item-card} Look up an API
:link: api/index
:link-type: doc

Task-oriented navigation plus generated, signature-level documentation for the
complete public Python surface.
:::
::::

# Analysis areas

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Linear mixed models
:class-card: pymixef-domain-card
:link: methods/lmm
:link-type: doc

**Continuous outcomes**

Gaussian ML/REML, multiple random blocks, conditional and population
predictions, residual diagnostics, simulation, and reproducible results.
:::

:::{grid-item-card} Generalized mixed models
:class-card: pymixef-domain-card
:link: methods/glmm
:link-type: doc

**Discrete outcomes**

Bernoulli, binomial, Poisson, and negative-binomial-2 reference fits using the
explicitly labeled first-order Laplace path.
:::

:::{grid-item-card} MMRM
:class-card: pymixef-domain-card
:link: methods/mmrm
:link-type: doc

**Repeated measures**

Ordered residual covariance, missing-response auditing, longitudinal contrasts,
and explicitly labeled degrees-of-freedom calculations.
:::

:::{grid-item-card} PK and ODE simulation
:class-card: pymixef-domain-card
:link: pharmacometrics/index
:link-type: doc

**Pharmacology**

Canonical dosing events, closed-form one- and two-compartment PK, event-aware
ODE integration, sensitivities, and residual-error models.
:::

:::{grid-item-card} Model declarations and IR
:class-card: pymixef-domain-card
:link: concepts/model-ir
:link-type: doc

**Inspectable models**

Formula and typed DSL authoring compile to a backend-neutral, immutable,
schema-versioned ModelIR with deterministic semantic hashes.
:::

:::{grid-item-card} Evidence and interchange
:class-card: pymixef-domain-card
:link: user-guide/results-provenance
:link-type: doc

**Reproducibility**

Structured convergence, diagnostics, archives, manifests, validation bundles,
comparison reports, and conservative external-format compatibility reports.
:::
::::

# Search

Open search with <kbd>Ctrl</kbd>/<kbd>⌘</kbd>+<kbd>K</kbd> or <kbd>/</kbd>.
Search accepts concepts such as “conditional prediction,” exact symbols such as
`Model.compile`, warning codes, CLI commands, and tutorial terms. When you open
a result, every matching phrase is highlighted on the page; use the match bar
or <kbd>Enter</kbd>/<kbd>Shift</kbd>+<kbd>Enter</kbd> to move between hits.

```{toctree}
:hidden:
:maxdepth: 2
:caption: Start

getting-started/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Learn

user-guide/index
Methods <methods/index>
pharmacometrics/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Examples

Tutorials <tutorials/index>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

Reference <reference/index>
API <api/index>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Trust & development

Evidence & development <trust/index>
```
