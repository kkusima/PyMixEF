(capability-catalog)=
# Capability catalog

This page is the human-readable view of the immutable
`pymixef.capabilities.CAPABILITIES` registry. It contains every registry
entry in this source tree: **56 unique capabilities**,
of which **42 are implemented** at their recorded scope and
**14 remain evidence-gated**. All entries currently carry the deliberately
conservative `experimental` maturity label.

Use the site search with a capability identifier, method name, evidence-file
path, or open-gate phrase. Each capability has its own anchored heading, and
its registry record expands to show exact state, evidence, and interpretation
boundaries.

## Start with a usable workflow

The registry is an evidence ledger, not the front door to an analysis. Choose
the workflow first, then use the identifier to inspect its exact boundary.

| Workflow | Implemented capability set | Main entry points |
|---|---|---|
| Gaussian hierarchical models | Dense ML/REML LMM, covariance construction and pathology reporting | `LMM-001`, `COV-001`, `COV-002`; {py:func}`pymixef.fit` |
| Grouped binary and count outcomes | Laplace Bernoulli, binomial, Poisson, and NB2 GLMM plus separation indicator | `GLMM-001`, `GLMM-003`, `DIST-001`–`DIST-003`; [GLMM guide](../methods/glmm.md) |
| Clinical repeated measures | Dense MMRM, covariance checks, and audited pattern-mixture delta adjustments | `MMRM-001`–`MMRM-003`; {py:func}`pymixef.data.pattern_mixture_adjust` |
| Pharmacometric simulation | Canonical events, deterministic ordering, reference ODEs, dose/event integration | `DATA-001`, `DATA-002`, `ODE-001`, `ODE-002`; [Events and ODEs](../pharmacometrics/events-and-ode.md) |
| Population-model building blocks | Conditional objectives/modes, transforms, BQL likelihoods, IOV representation | `NLME-001`–`NLME-004`; [Population estimation](../pharmacometrics/population-estimation.md) |
| Uncertainty and robustness checks | Restartable bootstrap, cross-setting refits, whole-group deletion influence | `INF-002`, `EST-003`, `DIAG-003`; {py:func}`pymixef.compare.approximation_sensitivity` |
| Reproducible evidence | Immutable IR, result archives, manifests, comparisons, validation bundles | `ARCH-001`, `API-002`, `PERF-002`, `REG-001`, `VAL-001`; [Validation](../validation.md) |

```{admonition} Interpret `implemented` at its recorded scope
:class: note

`implemented: true` means that the specifically named and scoped path exists
and has the evidence recorded below. It does not mean that the capability is
stable, reference-validated, suitable at every scale, or qualified for a
regulated workflow. Conversely, `implemented: false` can coexist with source
files or research helpers: the named integrated capability remains gated until
its stated evidence and workflow requirements are satisfied.
```

## Installation, representation, and execution are different

**Installation** answers whether the PyMixEF distribution and its selected
dependencies are present in an environment. Installing a module, command,
schema, or helper does not change a capability's evidence state.

**Representability** answers whether a construct can be expressed in ModelIR,
the formula layer, or the pharmacometrics DSL. A represented construct may
still be rejected by a selected engine. For example, the registry explicitly
records that not every distributional predictor represented by the IR is
executable by the initial Laplace backend; mixture weights and priors can also
be represented while their population-fitting or cross-sampler validation
gates remain open.

**Execution** answers whether a selected backend can perform the requested
calculation. The **evidence registry** goes further: it records the scoped
implementation state, maturity, reproducibility class, evidence, and open
gates. Check all four questions before relying on an analysis.

## Reading the fields

| Field | Meaning |
|---|---|
| Implemented | Exact registry Boolean. `true` is scoped to the named capability; `false` is an active evidence gate. |
| Stage | Program area used to organize delivery and evidence. It is not a maturity claim. |
| Maturity | Evidence tier. Every entry in this catalog is currently `experimental`. |
| Reproducibility | `bitwise`, `deterministic-with-tolerance`, `stochastic-with-monte-carlo-error`, or `None` when no executable guarantee is assigned. |
| Evidence summary | Exact test, source, schema, benchmark, or wrapper artifacts recorded in the registry. |
| Limitations / open gates | Scope boundary or remaining work recorded by the registry. |

Current reproducibility-class counts are:

- `bitwise`: 19;
- `deterministic-with-tolerance`: 21;
- `stochastic-with-monte-carlo-error`: 2; and
- no assigned class (`None`): 14.

## Query the live registry

From the command line:

```console
pymixef capabilities
pymixef capabilities --json
```

From Python:

```python
from pymixef import iter_capabilities

for capability in iter_capabilities(implemented=False):
    print(capability.identifier, capability.name)
    for gate in capability.limitations:
        print("  -", gate)
```

The CLI JSON form is best for automation. This page is optimized for
reading, browsing, and full-text search.

## Area overview

| Area | Registry prefix | Total | Implemented | Gated |
|---|---|---:|---:|---:|
| Architecture and backend contracts | `ARCH-*` | 3 | 2 | 1 |
| Public API and data behavior | `API-*` | 3 | 3 | 0 |
| Covariance | `COV-*` | 2 | 2 | 0 |
| Data and event semantics | `DATA-*` | 3 | 3 | 0 |
| Distribution contracts | `DIST-*` | 3 | 3 | 0 |
| Linear mixed models | `LMM-*` | 3 | 1 | 2 |
| Generalized mixed models | `GLMM-*` | 4 | 2 | 2 |
| Clinical longitudinal MMRM | `MMRM-*` | 3 | 3 | 0 |
| ODE integration | `ODE-*` | 3 | 2 | 1 |
| Nonlinear mixed effects | `NLME-*` | 5 | 4 | 1 |
| Population stochastic-approximation estimation | standalone ID | 1 | 0 | 1 |
| Estimation and convergence | `EST-*` | 3 | 2 | 1 |
| Inference | `INF-*` | 2 | 2 | 0 |
| Diagnostics | `DIAG-*` | 3 | 3 | 0 |
| Interoperability | `INT-*` | 2 | 1 | 1 |
| Advanced engines | `ADV-*` | 3 | 0 | 3 |
| Performance and reproducibility | `PERF-*` | 3 | 2 | 1 |
| Regulated workflow support | `REG-*` | 2 | 2 | 0 |
| Validation evidence | `VAL-*` | 3 | 3 | 0 |
| User experience and warnings | `UX-*` | 2 | 2 | 0 |

## Complete registry

Expand any record for its exact machine-readable fields and evidence notes.

## Architecture and backend contracts

Immutable model semantics, estimator compatibility, and shared backend conformance.

(capability-arch-001)=
### `ARCH-001` — Versioned immutable model IR

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `foundation` |
| Maturity | `experimental` |
| Reproducibility | `bitwise` |

**Evidence summary**

- Test evidence: `tests/test_ir.py`
- Schema evidence: `src/pymixef/schemas/model-ir-v1.json`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

(capability-arch-002)=
### `ARCH-002` — Estimator compatibility validation

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `foundation` |
| Maturity | `experimental` |
| Reproducibility | `bitwise` |

**Evidence summary**

- Test evidence: `tests/test_model.py`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

(capability-arch-003)=
### `ARCH-003` — Full backend conformance suite

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `false` |
| Stage | `foundation` |
| Maturity | `experimental` |
| Reproducibility | `None` (no class assigned) |

**Evidence summary**

- Implementation evidence: `src/pymixef/backends/base.py`
- Test evidence: `tests/test_backend_conformance.py`

**Limitations / open gates**

- A reusable fit-payload suite covers every built-in backend and checks validation, deterministic repeat fitting, input immutability, and row alignment. The blueprint's objective, gradient, optional Hessian-vector product, and simulation contracts are not yet part of the Backend Protocol.
```

## Public API and data behavior

Dry-run planning, result archival, and explicit row auditing.

(capability-api-001)=
### `API-001` — Dry-run compile, validate, and explain

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `foundation` |
| Maturity | `experimental` |
| Reproducibility | `bitwise` |

**Evidence summary**

- Test evidence: `tests/test_formula.py`
- Test evidence: `tests/test_model.py`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

(capability-api-002)=
### `API-002` — Stable result archival

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `foundation` |
| Maturity | `experimental` |
| Reproducibility | `bitwise` |

**Evidence summary**

- Test evidence: `tests/test_results.py`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

(capability-api-003)=
### `API-003` — Data audit without silent mutation

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `foundation` |
| Maturity | `experimental` |
| Reproducibility | `bitwise` |

**Evidence summary**

- Test evidence: `tests/test_data_covariance.py`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

## Covariance

Positive-definite parameterization and singularity reporting.

(capability-cov-001)=
### `COV-001` — Positive-definite covariance parameterizations

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `foundation` |
| Maturity | `experimental` |
| Reproducibility | `deterministic-with-tolerance` |

**Evidence summary**

- Test evidence: `tests/test_data_covariance.py`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

(capability-cov-002)=
### `COV-002` — Covariance singularity reporting

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `classical-core` |
| Maturity | `experimental` |
| Reproducibility | `deterministic-with-tolerance` |

**Evidence summary**

- Test evidence: `tests/test_lmm.py`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

## Data and event semantics

Reason-coded missingness plus canonical pharmacometric event records and ordering.

(capability-data-001)=
### `DATA-001` — Canonical pharmacometric event records

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `nlme-foundation` |
| Maturity | `experimental` |
| Reproducibility | `bitwise` |

**Evidence summary**

- Test evidence: `tests/test_events.py`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

(capability-data-002)=
### `DATA-002` — Deterministic same-time event ordering

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `nlme-foundation` |
| Maturity | `experimental` |
| Reproducibility | `bitwise` |

**Evidence summary**

- Test evidence: `tests/test_events.py`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

(capability-data-003)=
### `DATA-003` — Missingness contract and reason-coded audit

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `foundation` |
| Maturity | `experimental` |
| Reproducibility | `bitwise` |

**Evidence summary**

- Test evidence: `tests/test_data_covariance.py`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

## Distribution contracts

Likelihood normalization, family behavior, and distributional predictor representation.

(capability-dist-001)=
### `DIST-001` — Normalized likelihood policy

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `generalized-core` |
| Maturity | `experimental` |
| Reproducibility | `deterministic-with-tolerance` |

**Evidence summary**

- Test evidence: `tests/test_families.py`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

(capability-dist-002)=
### `DIST-002` — Family derivatives/CDF/simulation contract

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `generalized-core` |
| Maturity | `experimental` |
| Reproducibility | `deterministic-with-tolerance` |

**Evidence summary**

- Test evidence: `tests/test_families.py`

**Limitations / open gates**

- Analytic derivatives vary by family; finite-difference backend is used otherwise.
```

(capability-dist-003)=
### `DIST-003` — Distributional predictor representation

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `generalized-core` |
| Maturity | `experimental` |
| Reproducibility | `deterministic-with-tolerance` |

**Evidence summary**

- Test evidence: `tests/test_ir.py`
- Test evidence: `tests/test_families.py`

**Limitations / open gates**

- Not every represented predictor is executable by the initial Laplace backend.
```

## Linear mixed models

Gaussian ML/REML execution, scale targets, and inference gates.

(capability-lmm-001)=
### `LMM-001` — Gaussian LMM ML/REML reference engine

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `classical-core` |
| Maturity | `experimental` |
| Reproducibility | `deterministic-with-tolerance` |

**Evidence summary**

- Test evidence: `tests/test_lmm.py`
- Benchmark evidence: `benchmarks/b01_sleepstudy.json`

**Limitations / open gates**

- The 0.1 reference engine uses dense marginal covariance and is not the blueprint's million-row sparse production core.
```

(capability-lmm-002)=
### `LMM-002` — Sparse million-row LMM engine

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `false` |
| Stage | `classical-core` |
| Maturity | `experimental` |
| Reproducibility | `None` (no class assigned) |

**Evidence summary**

No completed evidence artifact is recorded for this gated capability.

**Limitations / open gates**

- Requires the planned compiled sparse backend.
```

(capability-lmm-003)=
### `LMM-003` — Profile, bootstrap, and robust LMM inference

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `false` |
| Stage | `classical-core` |
| Maturity | `experimental` |
| Reproducibility | `None` (no class assigned) |

**Evidence summary**

No completed evidence artifact is recorded for this gated capability.

**Limitations / open gates**

- A generic restartable bootstrap exists; profile and sandwich paths remain gated.
```

## Generalized mixed models

Laplace execution, approximation gates, separation diagnostics, and parity targets.

(capability-glmm-001)=
### `GLMM-001` — Laplace GLMM

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `generalized-core` |
| Maturity | `experimental` |
| Reproducibility | `deterministic-with-tolerance` |

**Evidence summary**

- Test evidence: `tests/test_glmm_mmrm.py`

**Limitations / open gates**

- Initial implementation targets documented low-dimensional random blocks.
```

(capability-glmm-002)=
### `GLMM-002` — Adaptive Gauss-Hermite quadrature

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `false` |
| Stage | `generalized-core` |
| Maturity | `experimental` |
| Reproducibility | `None` (no class assigned) |

**Evidence summary**

No completed evidence artifact is recorded for this gated capability.

**Limitations / open gates**

- Compatibility validation rejects AGHQ until an order-sensitivity suite lands.
```

(capability-glmm-003)=
### `GLMM-003` — Binary separation indicator

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `generalized-core` |
| Maturity | `experimental` |
| Reproducibility | `deterministic-with-tolerance` |

**Evidence summary**

- Test evidence: `tests/test_glmm_mmrm.py`

**Limitations / open gates**

- The indicator is a heuristic for Bernoulli/binomial fits; calibrated rare-event diagnostics and recovery benchmarks remain gated.
```

(capability-glmm-004)=
### `GLMM-004` — glmmTMB Salamanders parity

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `false` |
| Stage | `generalized-core` |
| Maturity | `experimental` |
| Reproducibility | `None` (no class assigned) |

**Evidence summary**

No completed evidence artifact is recorded for this gated capability.

**Limitations / open gates**

- Zero-inflated NB2 parity remains a later evidence gate.
```

## Clinical longitudinal MMRM

Repeated-measures covariance, inference labels, and missing-data sensitivity gates.

(capability-mmrm-001)=
### `MMRM-001` — MMRM REML with Satterthwaite and labeled approximate KR

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `clinical-longitudinal` |
| Maturity | `experimental` |
| Reproducibility | `deterministic-with-tolerance` |

**Evidence summary**

- Test evidence: `tests/test_glmm_mmrm.py`

**Limitations / open gates**

- The dense reference path is intended for small problems. Exact Kenward-Roger is rejected; only a clearly labeled approximation is available.
```

(capability-mmrm-002)=
### `MMRM-002` — MMRM covariance construction checks

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `clinical-longitudinal` |
| Maturity | `experimental` |
| Reproducibility | `deterministic-with-tolerance` |

**Evidence summary**

- Test evidence: `tests/test_data_covariance.py`
- Test evidence: `tests/test_glmm_mmrm.py`

**Limitations / open gates**

- Evidence covers positive-definite construction and local pathology tests, not external-software covariance-estimate conformance.
```

(capability-mmrm-003)=
### `MMRM-003` — Missing-data sensitivity transformations

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `clinical-longitudinal` |
| Maturity | `experimental` |
| Reproducibility | `deterministic-with-tolerance` |

**Evidence summary**

- Test evidence: `tests/test_sensitivity_workflows.py`

**Limitations / open gates**

- Applies audited additive response-scale deltas to explicitly identified, already-imputed cells. It does not perform imputation or choose clinically justified sensitivity deltas.
```

## ODE integration

Reference integration, dose/event semantics, and sensitivity-validation gates.

(capability-ode-001)=
### `ODE-001` — Reference ODE integration

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `nlme-foundation` |
| Maturity | `experimental` |
| Reproducibility | `deterministic-with-tolerance` |

**Evidence summary**

- Test evidence: `tests/test_ode_pk.py`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

(capability-ode-002)=
### `ODE-002` — Dose/event-aware ODE integration

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `nlme-foundation` |
| Maturity | `experimental` |
| Reproducibility | `deterministic-with-tolerance` |

**Evidence summary**

- Test evidence: `tests/test_events.py`
- Test evidence: `tests/test_ode_pk.py`

**Limitations / open gates**

- Covers the documented bolus, infusion, reset, and ADDL reference subset; steady-state event semantics are rejected.
```

(capability-ode-003)=
### `ODE-003` — Independently validated ODE sensitivities

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `false` |
| Stage | `nlme-foundation` |
| Maturity | `experimental` |
| Reproducibility | `None` (no class assigned) |

**Evidence summary**

- Implementation evidence: `src/pymixef/pharmacometrics/ode.py`
- Test evidence: `tests/test_ode_pk.py`

**Limitations / open gates**

- Forward and central finite-difference diagnostics plus one analytic decay check exist. Analytic/automatic sensitivities, event-discontinuity cases, and an independent multi-model validation suite do not.
```

## Nonlinear mixed effects

Conditional objectives, transforms, censoring, interoccasion variability, and mixture estimation.

(capability-nlme-001)=
### `NLME-001` — FOCEI-oriented conditional objective building blocks

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `nlme-foundation` |
| Maturity | `experimental` |
| Reproducibility | `deterministic-with-tolerance` |

**Evidence summary**

- Test evidence: `tests/test_pharmacometrics_dsl.py`

**Limitations / open gates**

- Provides subject objectives, conditional modes, and Laplace aggregation only. fit_focei() deliberately rejects because no integrated population optimizer or FitResult path exists.
```

(capability-nlme-002)=
### `NLME-002` — Population parameter transforms

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `nlme-foundation` |
| Maturity | `experimental` |
| Reproducibility | `deterministic-with-tolerance` |

**Evidence summary**

- Test evidence: `tests/test_ir.py`
- Test evidence: `tests/test_pharmacometrics_dsl.py`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

(capability-nlme-003)=
### `NLME-003` — Stable BQL/censoring likelihoods

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `pharmacometrics-breadth` |
| Maturity | `experimental` |
| Reproducibility | `deterministic-with-tolerance` |

**Evidence summary**

- Test evidence: `tests/test_ode_pk.py`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

(capability-nlme-004)=
### `NLME-004` — Interoccasion event representation

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `pharmacometrics-breadth` |
| Maturity | `experimental` |
| Reproducibility | `bitwise` |

**Evidence summary**

- Test evidence: `tests/test_events.py`

**Limitations / open gates**

- Population IOV estimation remains part of the production FOCEI gate.
```

(capability-nlme-005)=
### `NLME-005` — Finite-mixture population estimation

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `false` |
| Stage | `pharmacometrics-breadth` |
| Maturity | `experimental` |
| Reproducibility | `None` (no class assigned) |

**Evidence summary**

No completed evidence artifact is recorded for this gated capability.

**Limitations / open gates**

- Mixture weights can be represented; label-stable population fitting is unavailable.
```

## Population stochastic-approximation estimation

The standalone population-estimation gate recorded by the registry.

(capability-saem)=
### `SAEM` — Integrated SAEM population estimator

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `false` |
| Stage | `pharmacometrics-breadth` |
| Maturity | `experimental` |
| Reproducibility | `None` (no class assigned) |

**Evidence summary**

- Implementation evidence: `src/pymixef/pharmacometrics/estimation.py`
- Test evidence: `tests/test_pharmacometrics_dsl.py`

**Limitations / open gates**

- A callback-driven research kernel exists, but it is not connected to ModelIR, event/error-model compilation, population diagnostics, or the stable FitResult contract.
```

## Estimation and convergence

Unified convergence reporting, derivative verification, and approximation sensitivity.

(capability-est-001)=
### `EST-001` — Unified convergence object

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `foundation` |
| Maturity | `experimental` |
| Reproducibility | `bitwise` |

**Evidence summary**

- Test evidence: `tests/test_results.py`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

(capability-est-002)=
### `EST-002` — Independent derivative verification suite

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `false` |
| Stage | `foundation` |
| Maturity | `experimental` |
| Reproducibility | `None` (no class assigned) |

**Evidence summary**

- Implementation evidence: `src/pymixef/backends/base.py`
- Implementation evidence: `src/pymixef/pharmacometrics/estimation.py`
- Test evidence: `tests/test_pharmacometrics_dsl.py`
- Test evidence: `tests/test_lmm.py`

**Limitations / open gates**

- Finite-difference helpers are used for optimization diagnostics, but no separate derivative implementation and systematic cross-engine verification suite is available.
```

(capability-est-003)=
### `EST-003` — Approximation sensitivity workflow

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `generalized-core` |
| Maturity | `experimental` |
| Reproducibility | `deterministic-with-tolerance` |

**Evidence summary**

- Test evidence: `tests/test_sensitivity_workflows.py`

**Limitations / open gates**

- Deep-copied callback refits compare aligned parameters, covariance-derived standard errors, objectives, and caller-declared materiality flags. Scenario selection and thresholds are analyst-supplied and archived.
```

## Inference

Uncertainty provenance and restartable bootstrap support.

(capability-inf-001)=
### `INF-001` — Uncertainty provenance

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `foundation` |
| Maturity | `experimental` |
| Reproducibility | `bitwise` |

**Evidence summary**

- Test evidence: `tests/test_results.py`
- Implementation evidence: `src/pymixef/reporting.py`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

(capability-inf-002)=
### `INF-002` — Restartable bootstrap with failure accounting

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `classical-core` |
| Maturity | `experimental` |
| Reproducibility | `stochastic-with-monte-carlo-error` |

**Evidence summary**

- Test evidence: `tests/test_random_inference.py`

**Limitations / open gates**

- This is a generic callback/cluster bootstrap helper, not yet an integrated profile, BCa, or model-specific inference workflow.
```

## Diagnostics

Data-first diagnostics, VPC calculations, and influence-analysis gates.

(capability-diag-001)=
### `DIAG-001` — Diagnostic data-first contract

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `foundation` |
| Maturity | `experimental` |
| Reproducibility | `bitwise` |

**Evidence summary**

- Test evidence: `tests/test_results.py`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

(capability-diag-002)=
### `DIAG-002` — VPC calculations

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `pharmacometrics-breadth` |
| Maturity | `experimental` |
| Reproducibility | `stochastic-with-monte-carlo-error` |

**Evidence summary**

- Test evidence: `tests/test_results.py`

**Limitations / open gates**

- The 0.1 helper computes binned VPC tables from supplied simulations; it does not provide an integrated NLME simulation/refit workflow.
```

(capability-diag-003)=
### `DIAG-003` — Grouping-safe influence analysis

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `classical-core` |
| Maturity | `experimental` |
| Reproducibility | `deterministic-with-tolerance` |

**Evidence summary**

- Test evidence: `tests/test_sensitivity_workflows.py`

**Limitations / open gates**

- Full delete-group refits require archived fixed_effect_rank; Cook-style distance requires a finite symmetric positive-semidefinite baseline covariance. Optional approximations are reported beside, never substituted for, the full refit.
```

## Interoperability

Machine-readable compatibility reporting and the release-gated R wrapper.

(capability-int-001)=
### `INT-001` — Machine-readable compatibility reports

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `foundation` |
| Maturity | `experimental` |
| Reproducibility | `bitwise` |

**Evidence summary**

- Test evidence: `tests/test_interoperability.py`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

(capability-int-002)=
### `INT-002` — Release-gated thin R wrapper

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `false` |
| Stage | `foundation` |
| Maturity | `experimental` |
| Reproducibility | `None` (no class assigned) |

**Evidence summary**

- R-package evidence: `r/pymixef/R/pymixef.R`
- R-package evidence: `r/pymixef/man/pymixef-package.Rd`
- R-package evidence: `r/pymixef/tests/testthat/test-wrapper-api.R`
- R-package evidence: `r/pymixef/tests/testthat/test-python-parity.R`

**Limitations / open gates**

- The alpha reticulate package has Rd files and mocked/live testthat suites, R CMD build succeeds, and a dependency-complete local R CMD check --no-manual passes. No cross-platform R CI gate exists, and the maintainer address is an explicit non-routable placeholder.
```

## Advanced engines

Priors, robust sensitivity, and joint longitudinal-event targets.

(capability-adv-001)=
### `ADV-001` — Backend-neutral priors exported to two samplers

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `false` |
| Stage | `advanced-engines` |
| Maturity | `experimental` |
| Reproducibility | `None` (no class assigned) |

**Evidence summary**

No completed evidence artifact is recorded for this gated capability.

**Limitations / open gates**

- Priors are represented in the IR; two-backend equivalence is not yet validated.
```

(capability-adv-002)=
### `ADV-002` — Robust sensitivity comparison

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `false` |
| Stage | `advanced-engines` |
| Maturity | `experimental` |
| Reproducibility | `None` (no class assigned) |

**Evidence summary**

No completed evidence artifact is recorded for this gated capability.

**Limitations / open gates**

- No robust-likelihood fit path or automated cross-model sensitivity report is implemented.
```

(capability-adv-003)=
### `ADV-003` — Joint longitudinal-event simulation

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `false` |
| Stage | `advanced-engines` |
| Maturity | `experimental` |
| Reproducibility | `None` (no class assigned) |

**Evidence summary**

No completed evidence artifact is recorded for this gated capability.

**Limitations / open gates**

- No joint longitudinal/event model, shared random-effect simulator, or validated event-time likelihood is implemented.
```

## Performance and reproducibility

Benchmarking, reproducibility declarations, and numerical thread controls.

(capability-perf-001)=
### `PERF-001` — Reduced JSON benchmark harness

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `foundation` |
| Maturity | `experimental` |
| Reproducibility | `deterministic-with-tolerance` |

**Evidence summary**

- Benchmark evidence: `benchmarks/run.py`

**Limitations / open gates**

- The current harness contains one reduced synthetic LMM workload and is not the blueprint's cross-platform performance corpus.
```

(capability-perf-002)=
### `PERF-002` — Declared reproducibility classes

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `foundation` |
| Maturity | `experimental` |
| Reproducibility | `bitwise` |

**Evidence summary**

- Test evidence: `tests/test_provenance.py`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

(capability-perf-003)=
### `PERF-003` — Explicit numerical thread controls

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `false` |
| Stage | `foundation` |
| Maturity | `experimental` |
| Reproducibility | `None` (no class assigned) |

**Evidence summary**

- Implementation evidence: `src/pymixef/provenance.py`
- Test evidence: `tests/test_provenance.py`

**Limitations / open gates**

- Run manifests record ambient thread-related environment variables, but PyMixEF does not configure or enforce numerical library thread counts.
```

## Regulated workflow support

Validation-bundle and change-impact evidence helpers.

(capability-reg-001)=
### `REG-001` — Validation bundle generator

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `regulated-workflow-support` |
| Maturity | `experimental` |
| Reproducibility | `bitwise` |

**Evidence summary**

- Test evidence: `tests/test_validation.py`

**Limitations / open gates**

- Supports evidence generation; it is not a universal validation claim.
```

(capability-reg-002)=
### `REG-002` — Change-impact classification

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `regulated-workflow-support` |
| Maturity | `experimental` |
| Reproducibility | `bitwise` |

**Evidence summary**

- Test evidence: `tests/test_validation.py`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

## Validation evidence

Traceability, selected references, and the initial pathology corpus.

(capability-val-001)=
### `VAL-001` — Public traceability matrix

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `foundation` |
| Maturity | `experimental` |
| Reproducibility | `bitwise` |

**Evidence summary**

- Test evidence: `tests/test_validation.py`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

(capability-val-002)=
### `VAL-002` — Selected independent reference calculations

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `foundation` |
| Maturity | `experimental` |
| Reproducibility | `deterministic-with-tolerance` |

**Evidence summary**

- Test evidence: `tests/test_families.py`

**Limitations / open gates**

- External reference checks currently cover selected normalized family likelihoods; they are not independent full-engine or cross-software parity reports.
```

(capability-val-003)=
### `VAL-003` — Initial failure and pathology corpus

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `foundation` |
| Maturity | `experimental` |
| Reproducibility | `deterministic-with-tolerance` |

**Evidence summary**

- Test evidence: `tests/test_formula.py`
- Test evidence: `tests/test_data_covariance.py`
- Test evidence: `tests/test_glmm_mmrm.py`

**Limitations / open gates**

- The initial corpus covers representative parser, covariance, and engine failures; broad adversarial and platform-specific cases remain future work.
```

## User experience and warnings

Stable warning semantics and deterministic model comparison.

(capability-ux-001)=
### `UX-001` — Stable warning catalog

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `foundation` |
| Maturity | `experimental` |
| Reproducibility | `bitwise` |

**Evidence summary**

- Implementation evidence: `src/pymixef/warning_catalog.json`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

(capability-ux-002)=
### `UX-002` — Deterministic model diff

```{dropdown} Registry record

| Registry field | Exact value |
|---|---|
| Implemented | `true` |
| Stage | `foundation` |
| Maturity | `experimental` |
| Reproducibility | `bitwise` |

**Evidence summary**

- Test evidence: `tests/test_ir.py`

**Limitations / open gates**

No capability-specific limitation or open gate is recorded in the registry.
```

## Catalog-wide interpretation

```{admonition} A constructive way to use this catalog
:class: tip

Start with the scientific method you need, confirm that its scoped capability
is implemented, then read its reproducibility class, evidence, and open gates
together. For an unimplemented entry, the gate describes a concrete path to
stronger coverage rather than an invitation to bypass compatibility checks.
For an implemented experimental entry, use the evidence as a starting point
for design-specific verification and sensitivity work.
```
