# Executed tutorials

These ten notebooks are complete scientific walkthroughs, not fragments. Every
notebook is committed with outputs, assertions, and multiple plots; every page
below extracts the runnable core, exact saved results, figure-by-figure
interpretation, API map, exercises, and a constructive boundary around what the
analysis justifies.

All datasets are synthetic. That keeps the examples reproducible and safe to
share while leaving room for realistic clustered, longitudinal, dosing, and
model-evidence workflows.

## Materials and catalysis discovery

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} 01 · Screen catalyst candidates
:link: 01-catalyst-screening-lmm
:link-type: doc

**LMM · 120 runs · 3 figures**

Rank five candidates at a reference temperature while separating shared batch
variation, then compare conditional and population prediction.
:::

:::{grid-item-card} 02 · Model activation success
:link: 02-binary-catalyst-success-glmm
:link-type: doc

**Bernoulli GLMM · 96 runs · 3 figures**

Estimate promoter and temperature effects with catalyst-lot clustering,
conditional odds ratios, shrunken lot modes, and in-sample calibration.
:::

:::{grid-item-card} 03 · Analyze deactivation
:link: 03-catalyst-deactivation-mmrm
:link-type: doc

**MMRM · 144 observations · 3 figures**

Compare formulation-specific activity loss across six cycles using an explicitly
ordered AR(1) residual covariance.
:::
::::

## Bio, pharma, and medical analysis

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 04 · Multicenter biomarker trajectories
:link: 04-multicenter-biomarker-lmm
:link-type: doc

**LMM · center + patient random intercepts · 3 figures**

Separate multicenter and patient heterogeneity while interpreting a
treatment-by-week biomarker trajectory.
:::

:::{grid-item-card} 05 · Clinical-trial repeated measures
:link: 05-clinical-trial-mmrm
:link-type: doc

**MMRM · missing-response audit · 3 figures**

Fit change from baseline with AR(1) covariance, preserve four exclusions, and
construct adjusted treatment contrasts across visits.
:::

:::{grid-item-card} 06 · Clustered binary medical response
:link: 06-binary-response-glmm
:link-type: doc

**Bernoulli GLMM · 14 clinics · 3 figures**

Interpret cluster-conditional treatment odds, response curves, binned
calibration, and clinic random-intercept modes.
:::

:::{grid-item-card} 07 · Canonical dosing events
:link: 07-pharmacometrics-event-semantics
:link-type: doc

**Event semantics · provenance · 3 figures**

Canonicalize dosing and observation records, resolve same-time order, expand
ADDL and infusion stops, and audit every generated row.
:::

:::{grid-item-card} 08 · Closed-form PK and ODEs
:link: 08-closed-form-pk-and-ode
:link-type: doc

**PK/ODE simulation · sensitivities · 4 figures**

Cross-check an IV-bolus ODE against closed form, compare administration routes,
and decompose combined residual-error variance.
:::

:::{grid-item-card} 09 · Pharmacometric DSL and ModelIR
:link: 09-pharmacometrics-dsl-and-model-ir
:link-type: doc

**Typed model declaration · dependency graph · 3 figures**

Declare constrained parameters, ETAs, states, dosing, and observation error;
compile to an immutable ModelIR and verify deterministic round trips.
:::

:::{grid-item-card} 10 · Diagnostics and evidence lifecycle
:link: 10-diagnostics-simulation-validation-interop-archives
:link-type: doc

**LMM · simulation · archives/interchange · 3 figures**

Move from fit to residuals, VPC, reproducible simulation, integrity-checked
archives, validation bundles, and conservative compatibility reports.
:::
::::

## A suggested learning path

If you are new to mixed effects, start with 01, continue to 04, then compare 03
and 05 to understand LMM versus MMRM. Use 02 or 06 for link-scale GLMM
interpretation. The pharmacometrics sequence is deliberately progressive:

```text
07 event data → 08 numerical trajectories → 09 typed model contract
                                      ↓
                     10 evidence and interoperability
```

## Validation contract

The plots shown in these pages are byte-for-byte extracts of the pre-executed
notebook outputs. `scripts/extract_notebook_figures.py --check` verifies all 31
images, their cell/output source, dimensions, alt metadata, and SHA-256 hashes.
The notebook validator independently checks execution counts, errors, assertions,
output hygiene, figure presence, and accessibility metadata.

```{toctree}
:maxdepth: 1
:numbered:

01-catalyst-screening-lmm
02-binary-catalyst-success-glmm
03-catalyst-deactivation-mmrm
04-multicenter-biomarker-lmm
05-clinical-trial-mmrm
06-binary-response-glmm
07-pharmacometrics-event-semantics
08-closed-form-pk-and-ode
09-pharmacometrics-dsl-and-model-ir
10-diagnostics-simulation-validation-interop-archives
```

