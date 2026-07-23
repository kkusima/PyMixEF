# Fitting and convergence

A fit is interpretable only after the declared model, analysis rows, calculation
path, and numerical evidence have been reviewed.

## Engine and method

| Engine | Methods | Primary scope |
|---|---|---|
| `lmm` | `ml`, `reml` | Gaussian mixed models |
| `glmm` | `laplace` | Supported non-Gaussian mixed models |
| `mmrm` | `ml`, `reml` | Gaussian repeated measures with structured residual covariance |

The model validator checks family, link, covariance, and method compatibility.
It refuses unsupported requests such as AGHQ rather than substituting another
approximation.

## Explicit compile-and-fit

```python
model = pymixef.Model.from_formula("y ~ x + (1 | group)")
report = model.validate(data, engine="lmm", method="reml")
report.raise_for_errors()

plan = model.compile(
    data,
    engine="lmm",
    method="reml",
    maxiter=1000,
    tolerance=1e-10,
    compute_hessian=True,
)
print(plan.explain())
result = plan.fit()
```

Engine-specific accepted options are validated. Do not assume an option from
one backend applies to another.

## Structured convergence

`ConvergenceReport` records more than an optimizer boolean:

| Evidence | Question |
|---|---|
| optimizer state/message | Did the numerical routine terminate as intended? |
| objective and iteration/evaluation counts | What path and work produced the result? |
| raw and scaled gradients | Is the solution close to stationary on relevant scales? |
| `HessianDiagnostics` | Is local curvature available, symmetric, and positive definite? |
| `BoundaryRecord` entries | Are covariance or other parameters effectively on a boundary? |
| conditional-mode failures | Did any GLMM cluster-level inner optimization fail? |
| warning codes | Did a typed numerical/scientific condition occur? |
| optimizer sequence | Was a rescue/refinement path used? |
| `trustworthy` | Did the result pass the package’s combined interpretation gate? |

```python
report = result.convergence
print(report.to_dict())

if not report.trustworthy:
    for code in report.warning_codes:
        print(code)
```

## When the optional Hessian is disabled

`compute_hessian=False` can reduce reference-engine runtime, but the result
cannot claim Hessian-based uncertainty evidence that was not computed. Choose a
degrees-of-freedom/inference route consistent with available covariance
information and state the choice.

## Numerical review workflow

If a result is not trustworthy:

1. Preserve the exact result and warning codes.
2. Inspect design rank, group sizes, factor cells, visit coverage, and data
   exclusions.
3. Check starting scales, extreme covariates, and boundary variance estimates.
4. Simplify only with scientific justification.
5. Increase iteration limits or adjust tolerances when diagnostics support it.
6. Refit and compare parameters, objective, gradient, and warnings.
7. For GLMMs, inspect conditional-mode failures and approximation sensitivity.
8. Treat a materially changed specification as a new analysis, not merely a
   numerical repair.

## Reference implementation scope

The dense LMM, Laplace GLMM, and MMRM backends prioritize inspectability and
correctness experiments for small/moderate data. They are not the planned
compiled sparse million-row engine. Benchmark scalability and cross-software
parity for the exact model before consequential use.

## API map

- `Model.validate`, `Model.compile`, `ExecutionPlan.fit`
- `pymixef.backends.get_backend`, `fit_lmm`, `fit_glmm`, `fit_mmrm`
- `ConvergenceReport`, `HessianDiagnostics`, `BoundaryRecord`
- Stable errors in `pymixef.errors`; warning records in `pymixef.warnings`

