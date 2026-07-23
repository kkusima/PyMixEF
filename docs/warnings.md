# Warning and error catalog

The installed machine-readable catalog is `pymixef/warning_catalog.json`
(catalog version 1.0.0).
Structured records always include a stable code, severity, mathematical meaning,
likely causes, and remediation.

## Stable catalog

| Code | Severity | Meaning | Recommended review |
|---|---|---|---|
| `DATA-ROWS-EXCLUDED-001` | review | the missingness/validity contract removed source rows | reconcile every exclusion in `DataAudit` with the analysis plan |
| `DATA-FACTOR-LEVEL-DROPPED-001` | review | a declared factor level has no retained rows | review filtering and prespecified contrasts; do not silently choose a new reference |
| `DATA-DUPLICATE-KEY-001` | review | more than one row shares a repeated-measures key | add defining keys or resolve duplicate source records |
| `DATA-TIME-ORDER-001` | error | time decreases within a source group | normalize time scale and stable-sort within group |
| `FORMULA-RANK-DEFICIENT-001` | review | a fixed coefficient is not independently estimable | inspect aliased columns, empty cells, interactions, and collinearity |
| `FORMULA-CONSTANT-SCALE-001` | error | a constant predictor has zero scaling denominator | remove scaling or correct the predictor |
| `COV-BOUNDARY-001` | review | a variance component is at/near zero | inspect eigenvalues, design information, profiles, and justified simpler structures |
| `COV-SINGULAR-001` | review | a covariance block is rank deficient/nearly singular | inspect group counts, random design, and singularity report |
| `COV-CORRELATION-BOUNDARY-001` | review | correlation is near ±1 | inspect profiles and whether a diagonal structure is scientifically justified |
| `NUM-HESSIAN-INDEFINITE-001` | review | local curvature does not support ordinary Wald uncertainty | avoid naive Wald inference; inspect gradient, profiles, starts, and identification |
| `NUM-GRADIENT-LARGE-001` | review | optimizer termination did not reach the scaled-gradient tolerance | rescale, tighten controls, and compare starts/engines |
| `INFERENCE-BOUNDARY-LRT-001` | review | naive chi-square LRT is invalid at a variance boundary | use a documented mixture reference or parametric bootstrap |
| `ENGINE-APPROXIMATION-001` | information | the objective uses a named likelihood approximation | report method/controls and assess approximation sensitivity |
| `INTEROP-APPROXIMATED-001` | review | translation is only approximately equivalent | inspect compatibility and validate objective components |
| `SIM-MONTE-CARLO-ERROR-001` | information | simulation summary has material Monte Carlo error | increase deterministic-seed replicates and report Monte Carlo SE |

## Code families

- `DATA-*`: row exclusion, invalid keys, missing covariates, or event changes;
- `FORMULA-*`: unsafe, ambiguous, rank-deficient, or unsupported syntax;
- `COV-*`: boundary, singularity, or invalid covariance;
- `OPT-*`: optimizer termination or gradient concerns;
- `HESS-*`: indefinite or ill-conditioned Hessian;
- `ODE-*`: integration, event, or sensitivity failure;
- `ENGINE-*`: model/estimator incompatibility;
- `INTEROP-*`: transformed, approximated, or refused exchange semantics.

## Load, construct, and emit warnings

```python
from pymixef.warnings import (
    emit_warning,
    load_warning_catalog,
    warning_record,
)

catalog = load_warning_catalog()
record = warning_record("COV-SINGULAR-001")
emit_warning(record)
```

`PyMixEFWarning` is the base warning category.
`DataAuditWarning`, `CovarianceWarning`, and `NumericalWarning` allow standard
Python warning filters to target an operational area. A `WarningRecord` is
structured data; emitting it is a separate choice.

## Exception hierarchy

Errors that prevent calculation use `PyMixEFError` subclasses:

| Exception | Area |
|---|---|
| `ValidationError` | general public contract |
| `FormulaError` | formula syntax or compilation |
| `DataError` | data adaptation/audit |
| `CovarianceError` | covariance declaration/calculation |
| `TransformError` | parameter transform |
| `IRVersionError`, `IRValidationError` | ModelIR version/schema/semantics |
| `PluginError` | discovery or registry |
| `UnsupportedCapabilityError` | capability is not implemented |
| `EngineCompatibilityError` / `UnsupportedEngineError` | model/engine/method mismatch |
| `CompatibilityError` | strict external translation/report check |

Pharmacometric and backend modules define narrower subclasses documented on
their generated API pages.

A successful optimizer return does not suppress scientific or numerical warnings.
Use `fit.convergence.trustworthy`, not a raw success flag, when automating review.

The batch CLI returns exit code `4` when a fit completes with warning status.
Pass `--allow-warning` only when the surrounding workflow explicitly accepts
numerically suspect results. Failed fits return exit code `2`.

The complete signatures are in
[`pymixef.warnings`](api/generated/pymixef.warnings.rst) and
[`pymixef.errors`](api/generated/pymixef.errors.rst).
