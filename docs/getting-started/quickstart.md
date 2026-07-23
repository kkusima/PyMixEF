# Five-minute quickstart

This example fits a Gaussian random-intercept linear mixed model to repeated
measurements. It uses only the core installation.

## 1. Prepare columnar data

```python
import pymixef

data = {
    "change": [2.1, 3.2, 4.0, 2.4, 3.9, 5.0, 1.8, 2.9, 3.7],
    "time": [0, 1, 2, 0, 1, 2, 0, 1, 2],
    "subject": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
}
```

Mappings, NumPy arrays, pandas/Polars/Arrow-like frames, and xarray-like inputs
can all enter through the data adapter. Conversion is audited; source-row
identity is retained.

## 2. Declare the scientific model

```python
model = pymixef.Model.from_formula(
    "change ~ time + (1 | subject)",
    family=pymixef.families.Gaussian(),
)
```

The formula specifies:

- `change`: continuous response;
- `time`: population-average fixed slope;
- `(1 | subject)`: a Gaussian subject-specific random intercept;
- an implicit fixed intercept.

Use `0 +` or `- 1` to remove the fixed intercept. The formula parser does not
evaluate arbitrary Python.

## 3. Explain and compile before fitting

```python
print(model.explain(data, engine="lmm", method="reml"))

plan = model.compile(data, engine="lmm", method="reml")
print(plan.explain())
```

Review the formula expansion, fixed-design rank, random blocks, grouping levels,
missing-data audit, family/link, engine/method, covariance parameterization, and
ModelIR hash. Compilation creates an immutable execution plan; it does not
optimize model parameters.

## 4. Fit and gate interpretation on convergence

```python
result = plan.fit()

print(result.summary())
print(result.convergence.to_dict())
assert result.convergence.trustworthy
```

`trustworthy` combines optimizer status with numerical checks and warning state.
Never treat a returned parameter vector as sufficient evidence of a usable fit.

## 5. Ask a precise prediction question

```python
conditional = result.prediction(mode="conditional")
population = result.prediction(mode="population")
```

- **Conditional** predictions include estimated subject random effects and
  describe these fitted groups.
- **Population** predictions set random effects to their reference value and
  describe the fixed-effect mean.

Label the mode whenever predictions leave the analysis code.

## 6. Diagnose and preserve

```python
residual_table = result.residual_diagnostics()
print(residual_table.columns)

result.save("fit.json")
reloaded = pymixef.load("fit.json")

assert reloaded.parameters == result.parameters
assert reloaded.manifest.model_ir_hash == result.manifest.model_ir_hash
```

The JSON result is non-pickle and designed for inspection. `save` also creates
an integrity sidecar. A cryptographic hash detects accidental or unauthorized
change; it is not a digital signature.

## Compact convenience form

Once the workflow is familiar, `pymixef.fit` performs declaration, validation,
compilation, and fitting:

```python
result = pymixef.fit(
    "change ~ time + (1 | subject)",
    data=data,
    engine="lmm",
    method="reml",
)
```

The explicit `Model` → `ExecutionPlan` route is preferable when review,
debugging, or prospective validation matters.

## Where to go next

- [Core workflow](core-workflow.md) explains every boundary in the lifecycle.
- [Data and formulas](../user-guide/data-and-formulas.md) covers input and syntax.
- [LMM guide](../methods/lmm.md) explains likelihood and prediction semantics.
- [Tutorial 01](../tutorials/01-catalyst-screening-lmm.md) applies the workflow
  to catalyst screening with three figures and exact saved results.

