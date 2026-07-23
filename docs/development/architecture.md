# Architecture

PyMixEF’s central architectural rule is: represent scientific model meaning
once in a versioned, immutable intermediate representation, then let compatible
engines consume that representation.

## Layer map

| Layer | Modules | Responsibility |
|---|---|---|
| Public model | `model`, `formula` | builders, safe formula parsing, validation, compilation, dispatch |
| Data semantics | `data` | adapters, schema, row identity, missingness and reconciliation |
| Mathematical components | `families`, `covariance`, `transforms` | probability, link, covariance, and constraint contracts |
| Model representation | `ir` | typed nodes, schema validation/migration, semantic hashes, diffs |
| Numerical engines | `backends.*` | LMM, GLMM, MMRM reference calculations |
| Stable outputs | `results`, `convergence`, `diagnostics`, `provenance` | fit/result/evidence contracts |
| Inference/workflow | `inference`, `compare`, `reporting`, `validation`, `random` | bootstrap, parity, reports, bundles, deterministic streams |
| Pharmacometrics | `pharmacometrics.*` | DSL, event records, PK, ODE, population primitives |
| Exchange | `interoperability.*` | conservative import/export plus compatibility reports |
| Extensibility | `plugins`, `backends.base` | registries, discovery, and backend payload/protocol |

Semantic/compilation modules do not optimize. Backends do not redefine model
meaning. Result modules do not infer an unstated estimator.

## Shared backend payload

Backends return a validated mapping containing:

- natural-scale `parameters`;
- `unconstrained_parameters`;
- optional `parameter_covariance`;
- `fitted_values`, `residuals`, and `random_effects`;
- `objective`, `log_likelihood`, `method`, and `engine`;
- a mapping accepted by `ConvergenceReport.from_dict`;
- tidy column-oriented `diagnostic_data`;
- serializable engine-specific `extra` values.

`ExecutionPlan.fit()` turns this into a backend-neutral `FitResult` and joins it
to ModelIR, provenance, warnings, and audit state.

## ModelIR

`ModelIR` contains versioned nodes for parameters, fixed/random effects,
predictors, likelihoods, covariance, transforms, priors, state equations,
events, and outputs. Dependencies are explicit.

The IR boundary enables:

- model validation before numerical execution;
- deterministic JSON round trips and semantic hashes;
- structured model diffs and schema migration;
- multiple authoring interfaces;
- backend compatibility checks;
- conservative interoperability.

Representability does not imply that a current engine can estimate the model.

## Failure and warning policy

Backends raise typed `PyMixEFError` subclasses for invalid or incompatible
models. They never silently select a scientifically different estimator.
Completed calculations return structured convergence and stable warnings; a raw
optimizer success flag is not the interpretation gate.

## Extension points

Plugin registries exist for families, links, covariance, estimators,
diagnostics, exporters, and ODE solvers. A new numerical backend implements the
`Backend` protocol, accepts the common `CompiledData` payload, validates its
mapping, emits the shared result keys, declares reproducibility, and registers
static compatibility.

The reusable fit-contract suite exercises the shared payload, deterministic
repeat fitting, row alignment, and input immutability against every built-in
backend. Its case-coverage assertion fails when a built-in backend is
registered without a case. `ARCH-003` remains open: the public Backend Protocol
still needs separate objective, gradient, optional Hessian-vector product, and
simulation contracts before the blueprint's full conformance claim is met.

## Capability policy

Capabilities carry maturity, implemented state, evidence, limitations, and
reproducibility class. A partially specified or lower-level primitive can be
present without being declared an integrated production path. Gated estimator
entry points fail before optimization with stable unsupported codes and point to
available primitives where appropriate.

The source architecture file is also available for download:
{download}`ARCHITECTURE.md <../../ARCHITECTURE.md>`.
