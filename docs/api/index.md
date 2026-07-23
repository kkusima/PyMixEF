(api-reference)=
# API reference

This is the task-oriented entrance to the complete PyMixEF Python API. Every
public module is linked to its generated reference page, while the tables name
the classes, functions, constants, and protocols that readers are most likely
to search for directly.

Use the {doc}`generated module index <generated/modules>` when you already know
the import path. Use the {doc}`alias reference <aliases>` when code uses a
convenience name such as `pymixef.load`, `Normal`, `NB2`, `dry_run`, or
`estimated_marginal_means`.

```{toctree}
:hidden:

aliases
generated/modules
```

## Find an API by task

| Goal | Start here | Principal modules |
|---|---|---|
| Fit or inspect a model | {ref}`api-fit-model` | `pymixef`, `pymixef.model` |
| Parse a formula or build design matrices | {ref}`api-formulas` | `pymixef.formula` |
| Normalize, audit, or reconcile data | {ref}`api-data` | `pymixef.data` |
| Run LMM, GLMM, or MMRM calculations | {ref}`api-estimation-backends` | `pymixef.backends.*` |
| Calculate probability distributions | {ref}`api-families` | `pymixef.families` |
| Declare covariance or parameter constraints | {ref}`api-covariance-transforms` | `pymixef.covariance`, `pymixef.transforms` |
| Work with fit results, diagnostics, or uncertainty | {ref}`api-results-diagnostics` | `pymixef.results`, `pymixef.diagnostics`, `pymixef.inference` |
| Author a pharmacometric model | {ref}`api-pharmacometrics-authoring` | `pymixef.pharmacometrics.dsl` |
| Process dosing events, PK, or ODE systems | {ref}`api-events-pk-ode` | `pymixef.pharmacometrics.events`, `.pk`, `.ode`, `.estimation` |
| Inspect or migrate the model IR | {ref}`api-model-ir` | `pymixef.ir` |
| Translate external formats | {ref}`api-interoperability` | `pymixef.interoperability.*` |
| Capture evidence and reproducibility | {ref}`api-validation-reproducibility` | `pymixef.capabilities`, `.provenance`, `.random`, `.validation` |
| Register extensions | {ref}`api-plugins` | `pymixef.plugins`, `pymixef.backends.base` |
| Handle errors, warnings, native discovery, or the CLI | {ref}`api-operational` | `pymixef.errors`, `.warnings`, `.native`, `.cli` |

## Root exports and submodule APIs

The {doc}`pymixef root module <generated/pymixef>` is the concise everyday
surface. It exports `fit`, `Model`, `ModelIR`, `FitResult`, `compare`,
`bootstrap`, `render_report`, capability and validation helpers, convergence
types, stable exception types, and the `families`, `covariance`, `diagnostics`,
`interoperability`, and `pharmacometrics` namespaces.

Specialized APIs remain in their owning modules:

```python
import pymixef
from pymixef.data import audit_data
from pymixef.formula import compile_formula
from pymixef.pharmacometrics import canonicalize_events, simulate_ode

result = pymixef.fit("response ~ time + (1 | subject)", data=data)
audited = audit_data(data, response="response", covariates=("time",))
matrices = compile_formula("response ~ time", data)
```

An object can therefore be public without being a direct `pymixef.<name>`
export. For example, `DiagnosticTable` belongs to `pymixef.diagnostics`,
`Capability` belongs to `pymixef.capabilities`, and `CompatibilityReport`
belongs to `pymixef.interoperability`. Prefer the documented owning module for
stable imports. See {doc}`aliases` for names that deliberately resolve to
another public object.

:::{important} Catalog availability is not estimator support
The family, covariance, IR, and pharmacometric catalogs are broader than the
formula backends that can execute a complete fit.

- Formula LMM execution is Gaussian ML or REML.
- Formula GLMM execution is the documented Bernoulli, binomial, Poisson, and
  negative-binomial-2 Laplace subset.
- The dedicated MMRM path is Gaussian REML with an explicit residual covariance.
- Other `Family` classes remain useful for normalized likelihood, probability,
  moment, and simulation calculations, but their presence does not make them
  executable by `pymixef.fit`.
- Priors and distributional predictors can be represented in `ModelIR`; current
  formula backends do not execute those declarations.
- `fit_focei()` deliberately rejects because an integrated production FOCEI
  path is not present. `experimental_saem()` is a callback-driven experimental
  kernel rather than an integrated population estimator.

Check `pymixef.iter_capabilities()` or `pymixef capabilities --json` before
depending on an experimental calculation path.
:::

(api-fit-model)=
## 1. Fit, compile, validate, and explain a model

| Module | Public objects |
|---|---|
| {doc}`pymixef <generated/pymixef>` | `fit`, `Model`, `Response`, `Fixed`, `Random`, `ExecutionPlan`, `FitResult`, `load` |
| {doc}`pymixef.model <generated/pymixef.model>` | Classes `Response`, `Fixed`, `Random`, `ValidationFinding`, `ValidationReport`, `Model`, `ExecutionPlan`; function `fit` |

`Model.from_formula()` creates a backend-neutral model. `Model.validate()`,
`Model.explain()`, and `Model.compile()` inspect compatibility before
optimization. `ExecutionPlan.validate()`, `ExecutionPlan.explain()`, and
`ExecutionPlan.to_backend_data()` expose the deterministic compiled plan;
`ExecutionPlan.fit()` executes it. `pymixef.fit()` is the direct formula or
prebuilt-`Model` entry point.

(api-formulas)=
## 2. Formulas and design matrices

| Module | Public classes | Public functions |
|---|---|---|
| {doc}`pymixef.formula <generated/pymixef.formula>` | `RandomTerm`, `FormulaSpec`, `RandomDesignBlock`, `FormulaExplanation`, `DesignMatrices` | `parse_formula`, `compile_formula`, `dry_run`, `model_matrix`, `explain_formula` |

Search for `FormulaSpec.to_ir`, `FormulaSpec.explain`,
`DesignMatrices.to_backend_data`, `DesignMatrices.explanation`, and
`DesignMatrices.explain` when inspecting compilation. Formula expressions
support documented fixed effects, interactions, nesting, correlated `|` and
independent `||` random blocks, categorical encoding, safe transforms, and
row-complete audit metadata.

(api-data)=
## 3. Data adaptation, missingness, and audit

| Module | Public classes and enums | Public functions |
|---|---|---|
| {doc}`pymixef.data <generated/pymixef.data>` | `MissingnessKind`, `ColumnSchema`, `ColumnarData`, `DataAdapter`, `AuditRecord`, `DataAudit`, `AuditedData`, `PatternMixtureRecord`, `PatternMixtureResult` | `is_missing`, `missing_mask`, `adapt_data`, `InputAdapter`, `audit_data`, `prepare_data`, `pattern_mixture_adjust`, `find_duplicate_keys`, `validate_monotonic_time`, `stable_sort` |

`ColumnarData` is the immutable internal table and exposes `n_rows`,
`n_columns`, `column_names`, `take`, `to_dict`, and `fingerprint`. `DataAudit`
exposes source-to-analysis reconciliation through `excluded_rows`,
`excluded_row_ids`, `reason_counts`, and `to_dict`.

(api-estimation-backends)=
## 4. LMM, GLMM, and MMRM backends

| Module | Public API |
|---|---|
| {doc}`pymixef.backends <generated/pymixef.backends>` | `BUILTIN_BACKENDS`, `Backend`, `BackendError`, `BackendInputError`, `BackendUnsupportedError`, `BackendNumericalError`, `GaussianLMMBackend`, `LMMBackend`, `DenseLMMBackend`, `LaplaceGLMMBackend`, `GLMMBackend`, `MMRMBackend`, `get_backend`, `fit_lmm`, `fit_glmm`, `fit_mmrm` |
| {doc}`pymixef.backends.lmm <generated/pymixef.backends.lmm>` | `GaussianLMMBackend`, `LMMBackend`, `DenseLMMBackend`, `fit_lmm` |
| {doc}`pymixef.backends.glmm <generated/pymixef.backends.glmm>` | `LaplaceGLMMBackend`, `GLMMBackend`, `fit_glmm` |
| {doc}`pymixef.backends.mmrm <generated/pymixef.backends.mmrm>` | `MMRMBackend`, `fit_mmrm`, `linear_inference`, `estimated_marginal_means`, `contrasts` |
| {doc}`pymixef.backends.base <generated/pymixef.backends.base>` | `Backend`, backend error classes, `RandomBlockData`, `CompiledData`, `CovarianceParameterization`, `field`, `backend_mapping`, `factorize`, `prepare_random_block`, `prepare_data`, `covariance_slices`, `random_covariance`, `safe_cholesky`, `cho_solve`, `logdet_from_cholesky`, `finite_hessian`, `finite_gradient`, `covariance_from_hessian`, `convergence_mapping`, `make_payload`, `validate_payload` |

The backend functions return the lower-level payload contract.
`pymixef.fit()` and `ExecutionPlan.fit()` convert that payload into a
`FitResult` with convergence, provenance, warnings, diagnostics, and archived
model semantics.

(api-families)=
## 5. Probability families and links

{doc}`pymixef.families <generated/pymixef.families>` defines the common
`Family` and `Link` contracts.

| Catalog | Public objects |
|---|---|
| Links | `Link`, `links`, `get_link`, `IDENTITY`, `LOG`, `LOGIT`, `PROBIT`, `CLOGLOG`, `CAUCHIT`, `INVERSE`, `INVERSE_SQUARED` |
| Continuous | `Gaussian`, `StudentT`, `LogNormal`, `Gamma`, `InverseGaussian`, `Beta`, `Tweedie` |
| Discrete and categorical | `Bernoulli`, `Binomial`, `Poisson`, `NegativeBinomial1`, `NegativeBinomial2`, `GeneralizedPoisson`, `COMPoisson`, `Ordinal`, `Multinomial` |
| Mixture and observation wrappers | `ZeroInflated`, `Hurdle`, `Truncated`, `Censored` |
| Survival | `Exponential`, `LogNormalSurvival`, `Weibull`, `Gompertz`, `LogLogistic`, `PiecewiseExponential` |

The shared calculation vocabulary is `log_prob`, `log_probability`, `logpdf`,
`logpmf`, `cdf`, `logcdf`, `sf`, `logsf`, `rvs`, `random`, `mean`, `variance`,
and `moments`. The {doc}`alias reference <aliases>` lists short names including
`Normal`, `NB1`, `NB2`, `GenPoisson`, and the `*Family` and `*Survival` aliases.

(api-covariance-transforms)=
## 6. Covariance structures and parameter transforms

| Module | Public API |
|---|---|
| {doc}`pymixef.covariance <generated/pymixef.covariance>` | `CovarianceValidation`, `CovarianceStructure`, `Diagonal`, `Unstructured`, `CompoundSymmetry`, `AR1`, `HeterogeneousAR1`, `Toeplitz`, `HeterogeneousToeplitz`, `AnteDependence`, `SpatialPower`, `KnownCovariance`, `validate_covariance`, `covariance_structure`, `get_covariance`, `singularity_report`, and the documented `*Covariance` aliases |
| {doc}`pymixef.transforms <generated/pymixef.transforms>` | `Transform`, `IdentityTransform`, `LogTransform`, `SoftplusTransform`, `BoundedTransform`, `SimplexTransform`, `OrderedTransform`, `CholeskyCovarianceTransform`, `get_transform` |

`CovarianceStructure.parameter_count`, `parameter_names`, `covariance`,
`matrix`, `validate`, `derivatives`, `simulate`, and `to_dict` form the
structure contract. `Transform.forward`, `inverse`,
`log_abs_det_jacobian`, and `jacobian` form the optimizer-to-natural-scale
contract.

(api-results-diagnostics)=
## 7. Results, diagnostics, simulation, inference, and reporting

| Module | Public API |
|---|---|
| {doc}`pymixef.results <generated/pymixef.results>` | `FitResult`; properties `success`, `n_observations`; methods `prediction`, `diagnostic`, `residual_diagnostics`, `simulate`, `vpc`, `to_dict`, `save`, `from_dict`, `load`, `summary` |
| {doc}`pymixef.diagnostics <generated/pymixef.diagnostics>` | `DiagnosticTable`, `GroupInfluenceResult`, `residual_table`, `vpc_table`, `covariance_singularity_table`, `group_influence` |
| {doc}`pymixef.inference <generated/pymixef.inference>` | `BootstrapResult`, `bootstrap` |
| {doc}`pymixef.compare <generated/pymixef.compare>` | `ComparisonResult`, `ApproximationSensitivityResult`, `compare`, `approximation_sensitivity` |
| {doc}`pymixef.reporting <generated/pymixef.reporting>` | `render_report` |
| {doc}`pymixef.convergence <generated/pymixef.convergence>` | `HessianDiagnostics`, `BoundaryRecord`, `ConvergenceReport` |

`DiagnosticTable.save()` and `DiagnosticTable.load()` handle JSON and CSV
diagnostic data. `BootstrapResult.intervals()` provides the documented portable
percentile interval path. `GroupInfluenceResult` retains group-level refit and
approximation failures; `ApproximationSensitivityResult` retains exact named
settings and cross-refit failures. `ComparisonResult.assert_within()` and
`ComparisonResult.write_report()` support explicit cross-platform tolerances
and reports. `render_report()` emits Markdown, HTML, PDF, or Word according to
the destination suffix and installed optional dependencies.

(api-pharmacometrics-authoring)=
## 8. Pharmacometric model authoring

The {doc}`pymixef.pharmacometrics umbrella
<generated/pymixef.pharmacometrics>` re-exports the typed pharmacometric
surface. The owning authoring module is
{doc}`pymixef.pharmacometrics.dsl <generated/pymixef.pharmacometrics.dsl>`.

| Area | Public objects |
|---|---|
| Expressions and symbols | `Expr`, `Param`, `Eta`, `State`, `Symbol`, `as_expr`, `symbol`, `covariate`, `exp`, `log`, `sqrt`, `log1p` |
| Differential equations and observations | `Dose`, `DifferentialEquation`, `derivative`, `d`, `Observation`, `observe` |
| Model declarations | `ValidationMessage`, `ModelValidation`, `CompiledModel`, `ModelDefinition`, `model`, `compiled_model` |
| Error | `DSLValidationError` |

Search for `Param.real`, `Param.positive`, `Param.bounded`, `Eta.correlated`,
`Eta.independent`, `Dose.into`, `State.derivative`, `Expr.evaluate`,
`CompiledModel.to_ir`, and `ModelDefinition.compile` for the principal builder
operations.

(api-events-pk-ode)=
## 9. Events, closed-form PK, ODE simulation, and population primitives

| Module | Public API |
|---|---|
| {doc}`pymixef.pharmacometrics.events <generated/pymixef.pharmacometrics.events>` | `EventValidationError`, `EventType`, `DoseAmountStatus`, `EVENT_PRIORITY`, `AuditEntry`, `CanonicalEvent`, `EventTable`, `canonicalize_events` |
| {doc}`pymixef.pharmacometrics.pk <generated/pymixef.pharmacometrics.pk>` | `PKValidationError`, `OneCompartmentPK`, `TwoCompartmentPK`, `TwoCompartmentRates`, `one_compartment_iv_bolus`, `one_compartment_infusion`, `one_compartment_oral`, `two_compartment_rates`, `two_compartment_iv_bolus`, `two_compartment_infusion`, `two_compartment_oral`, `ObservationError`, `AdditiveError`, `ProportionalError`, `PowerError`, `CombinedError`, `LogNormalError`, `additive`, `proportional`, `power`, `combined`, `lognormal`, `left_censored_loglikelihood`, `right_censored_loglikelihood`, `interval_censored_loglikelihood`, and PK route-name aliases |
| {doc}`pymixef.pharmacometrics.ode <generated/pymixef.pharmacometrics.ode>` | `ODESimulationError`, `UnsupportedEventSemantics`, `ODEContext`, `EventSnapshot`, `ODESolverMetadata`, `ODESimulationResult`, `SensitivityCheck`, `simulate_ode`, `finite_difference_sensitivities`, `simulate_subjects` |
| {doc}`pymixef.pharmacometrics.estimation <generated/pymixef.pharmacometrics.estimation>` | `EstimationError`, `UnsupportedEstimatorError`, `ConditionalModeError`, `SAEMError`, `ObjectiveComponents`, `ConditionalObjective`, `ConditionalModeResult`, `LaplacePopulationResult`, `SAEMProblem`, `SAEMControl`, `SAEMResult`, `apply_random_effects`, `omega_from_standard_deviations`, `eta_shrinkage`, `conditional_mode_objective`, `finite_difference_gradient`, `finite_difference_hessian`, `find_conditional_mode`, `laplace_population_objective`, `fit_focei`, `experimental_saem`, `saem` |

`CanonicalEvent` exposes event classification, effective dose/rate/duration, and
record conversion. `EventTable` provides subject selection, source and
canonical record conversion, ADDL and infusion expansion, and provenance.
`ODESimulationResult.state()` and `sensitivity()` select named output arrays.

(api-model-ir)=
## 10. Model IR, schema migration, hashing, and diffing

{doc}`pymixef.ir <generated/pymixef.ir>` defines:

- schema constants `IR_SCHEMA_VERSION`, `MODEL_IR_SCHEMA_VERSION`,
  `LEGACY_IR_SCHEMA_VERSION`, and `SUPPORTED_IR_SCHEMA_VERSIONS`;
- `IRNode` and the typed nodes `ParameterIR`, `FixedEffectIR`,
  `RandomEffectIR`, `PredictorIR`, `LikelihoodIR`, `CovarianceIR`,
  `TransformIR`, `PriorIR`, `StateEquationIR`, `EventIR`, and `OutputIR`;
- the complete `ModelIR` graph;
- `register_ir_migration`, `migrate_ir`, `diff_models`, and alias `model_diff`;
- `DiffEntry` and `ModelDiff`.

`ModelIR.to_dict()`, `canonical_json()`, `to_json()`, `from_dict()`,
`from_json()`, `semantically_equal()`, and `diff()` are the serialization and
comparison operations. `ModelIR.semantic_hash` and `ModelIR.hash` are
properties. `ModelDiff.equal` and `ModelDiff.categories` are also properties.

(api-interoperability)=
## 11. Interoperability and compatibility accounting

| Module | Public API |
|---|---|
| {doc}`pymixef.interoperability <generated/pymixef.interoperability>` | `CompatibilityReport`, `InterchangeResult`, all public import, export, parse, and translation functions below |
| {doc}`pymixef.interoperability.base <generated/pymixef.interoperability.base>` | `CompatibilityReport`, `InterchangeResult`, `issues` |
| {doc}`pymixef.interoperability.nonmem <generated/pymixef.interoperability.nonmem>` | `parse_control_stream`, `import_nonmem_data`, `import_nonmem_table` |
| {doc}`pymixef.interoperability.pharmml <generated/pymixef.interoperability.pharmml>` | `import_pharmml`, `export_pharmml` |
| {doc}`pymixef.interoperability.sbml <generated/pymixef.interoperability.sbml>` | `import_sbml`, `export_sbml` |
| {doc}`pymixef.interoperability.sedml <generated/pymixef.interoperability.sedml>` | `import_sedml`, `export_sedml` |
| {doc}`pymixef.interoperability.r <generated/pymixef.interoperability.r>` | `translate_r_formula` |

Every interchange function returns a value with explicit compatibility
accounting. Use `CompatibilityReport.supported`, `by_status`,
`require_supported`, `to_dict`, and `write`; use
`InterchangeResult.require_supported()` when unsupported constructs must stop a
workflow.

(api-validation-reproducibility)=
## 12. Capabilities, provenance, randomness, and validation evidence

| Module | Public API |
|---|---|
| {doc}`pymixef.capabilities <generated/pymixef.capabilities>` | `Capability`, `CAPABILITIES`, `get_capability`, `iter_capabilities` |
| {doc}`pymixef.provenance <generated/pymixef.provenance>` | `environment_snapshot`, `fingerprint_data`, `fingerprint_model_ir`, `RunManifest`, `RunTimer` |
| {doc}`pymixef.random <generated/pymixef.random>` | `RandomStreamManager`, `random_streams` |
| {doc}`pymixef.validation <generated/pymixef.validation>` | `RequirementLinks`, `TRACEABILITY_LINKS`, `TraceabilityRecord`, `traceability_matrix`, `classify_change`, `change_impact`, `create_validation_bundle`, `verify_validation_bundle` |
| {doc}`pymixef root <generated/pymixef>` | `Maturity`, `ReproducibilityClass`, `WarningRecord` |

`RunManifest.capture()` records model and data fingerprints, environment,
settings, seeds, reproducibility class, convergence, and warnings.
`RandomStreamManager.generator()` and `replicates()` derive order-independent
Philox streams. Validation bundles are created by
`create_validation_bundle()` and checked by `verify_validation_bundle()`.

(api-plugins)=
## 13. Plugins and advanced backend contracts

{doc}`pymixef.plugins <generated/pymixef.plugins>` exposes:

- `PluginInfo` and generic `Registry`;
- protocols `CovariancePlugin` and `EstimatorPlugin`;
- `discover_plugins`;
- `FAMILY_REGISTRY`, `LINK_REGISTRY`, `COVARIANCE_REGISTRY`,
  `ESTIMATOR_REGISTRY`, `DIAGNOSTIC_REGISTRY`, `EXPORTER_REGISTRY`, and
  `ODE_SOLVER_REGISTRY`.

Use `Registry.register`, `unregister`, `get`, `info`, `names`, and `snapshot`
for explicit runtime registration. Low-level implementers should also consult
{doc}`pymixef.backends.base <generated/pymixef.backends.base>` for the `Backend`
protocol and compiled numeric payload contracts.

(api-operational)=
## 14. Errors, warnings, native discovery, and command line

| Module | Public API |
|---|---|
| {doc}`pymixef.errors <generated/pymixef.errors>` | `PyMixEFError`, `ValidationError`, `FormulaError`, `DataError`, `CovarianceError`, `TransformError`, `IRVersionError`, `IRValidationError`, `PluginError`, `UnsupportedCapabilityError`, `EngineCompatibilityError`, `UnsupportedEngineError`, `CompatibilityError` |
| {doc}`pymixef.warnings <generated/pymixef.warnings>` | `PyMixEFWarning`, `DataAuditWarning`, `CovarianceWarning`, `NumericalWarning`, `load_warning_catalog`, `warning_record`, `emit_warning` |
| {doc}`pymixef.native <generated/pymixef.native>` | `library_path`, `native_available`, `core_version` |
| {doc}`pymixef.cli <generated/pymixef.cli>` | programmatic function `main`; console commands `capabilities`, `traceability`, `explain`, `fit`, `bundle`, `verify-bundle`, `parse-nonmem` |
| {doc}`pymixef root <generated/pymixef>` | `__version__` and root-exported error classes |

Expected PyMixEF exceptions carry `code`, `remediation`, `details`, and
`source_location`; use `PyMixEFError.to_dict()` instead of parsing display
messages. Structured warnings carry a `WarningRecord`, and
`load_warning_catalog()` returns the packaged stable warning catalog.
