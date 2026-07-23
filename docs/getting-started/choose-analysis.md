# Choose an analysis

Start with the endpoint and dependence structure, then ask what quantity you
need to estimate. Software availability comes after the scientific question.

## Decision table

| Scientific situation | Starting point | Typical declaration | Current calculation path |
|---|---|---|---|
| Continuous outcome; observations clustered in batches, sites, people, or devices | Gaussian LMM | `y ~ treatment + time + (1 | group)` | ML or REML |
| Continuous outcome; subject-specific trajectories | Gaussian LMM | `y ~ treatment * time + (1 + time | subject)` | ML or REML |
| Binary, count, or overdispersed-count outcome with cluster effects | GLMM | `y ~ treatment + (1 | site)` plus family | First-order Laplace |
| Repeated continuous visits where residual covariance is the focus | MMRM | fixed-effects formula plus `AR1`, `Toeplitz`, or `Unstructured` residual | ML or REML |
| Known-parameter one-/two-compartment concentration profile | Closed-form PK | `one_compartment_*`, `two_compartment_*` | Direct calculation |
| Dosing events, discontinuities, custom dynamics, or sensitivities | Event-aware ODE | canonical event table plus `simulate_ode` | Numerical integration |
| Auditable population-PK model contract | Pharmacometric DSL | parameters, ETAs, states, doses, observations | Compile/validate/IR; simulation and conditional-mode primitives |
| Cross-tool model/data exchange | Interoperability layer | NONMEM, PharmML, SBML, SED-ML, R-formula helpers | Supported subset with compatibility report |

## LMM or MMRM?

Both can analyze continuous longitudinal outcomes, but their dependence models
answer different modeling needs.

- Choose an **LMM** when random effects are the natural representation of
  between-group heterogeneity or subject-specific trajectories.
- Choose **MMRM** when an explicit within-subject residual covariance across a
  finite visit axis is central to the analysis.
- The current reference MMRM path does not combine its structured residual
  covariance with formula random effects.

## LMM or GLMM?

Choose from the conditional distribution of the observed endpoint.

- Approximately Gaussian continuous response → Gaussian LMM.
- Binary response → Bernoulli/binomial GLMM with logit link.
- Count response → Poisson GLMM when mean and variance assumptions are
  reasonable; negative-binomial-2 is available for a supported overdispersion
  path.

PyMixEF defines a broader catalog of probability-family objects for standalone
probability calculations and model representation. Catalog presence does not
mean every family can be fitted by the formula GLMM backend. The current GLMM
engine fits Bernoulli, binomial, Poisson, and negative-binomial-2 with canonical
links and first-order Laplace only.

## Formula model or pharmacometric DSL?

Use the formula interface for LMM, GLMM, and MMRM workflows. Use the typed
pharmacometric DSL when the model contains parameters with constraints, random
effects on transformed parameters, states, differential equations, dose
mappings, and observation models.

Both routes compile to a versioned `ModelIR`. The same representation makes
capability checking, semantic hashing, comparison, serialization, and backend
selection inspectable.

## Estimation or simulation?

Closed-form PK helpers and `simulate_ode` calculate trajectories from supplied
parameters; they do not estimate population parameters. The DSL declares and
validates a model contract. Population-estimation primitives expose conditional
mode and Laplace calculations, while integrated production FOCEI and SAEM
workflows are not currently available.

## Before committing to a method

Ask:

1. What is the observational unit and what creates dependence?
2. Is the endpoint continuous, binary, count, censored, or time-to-event?
3. Is the target population-average, cluster-conditional, subject-specific, or
   a contrast at a particular visit?
4. Which missing-data assumption is scientifically defensible?
5. Is visit order, dosing order, or another time axis part of the model state?
6. Which covariance alternatives and diagnostics will be prespecified?
7. Does the [validation record](../validation.md) provide enough evidence for
   the intended consequence?

Use the [full analysis matrix](../reference/analysis-matrix.md) for supported
methods, families, outputs, and explicit boundaries.
