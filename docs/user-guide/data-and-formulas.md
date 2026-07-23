# Data and formulas

PyMixEF accepts familiar tabular inputs but compiles them into its own immutable
columnar representation. This keeps data conversion, factor levels, missingness,
and source-row reconciliation explicit.

## Supported input shapes

`pymixef.data.adapt_data` recognizes:

- mappings from column names to one-dimensional values;
- plain two-dimensional NumPy arrays when `column_names` is supplied;
- structured NumPy arrays;
- pandas-like and Polars-like dataframes;
- Arrow-like tables;
- xarray-like objects that expose a dataframe conversion.

The adapter does not require optional dataframe libraries merely to import
PyMixEF. Install the `data` extra when those libraries are part of your workflow.

```python
from pymixef.data import adapt_data

table = adapt_data(
    {"y": [1.1, 1.7], "subject": ["A", "B"]},
    roles={"y": "response", "subject": "group"},
    units={"y": "mg/L"},
)

print(table.n_rows, table.column_names)
print(table.schema)
```

`ColumnarData` provides mapping-like access, stable row IDs, source indices,
schema entries, subsetting, and a copy-controlled `to_dict`.

## Missingness is a contract

`audit_data` distinguishes missing response, missing covariate, censored
response, structurally absent endpoint, and invalid record. The policy
`missing="drop"`, `"raise"`, or `"keep"` controls ordinary missing values;
invalid and structurally absent records remain excluded.

```python
from pymixef.data import audit_data

audited = audit_data(
    data,
    response="change",
    covariates=("baseline", "visit", "subject"),
    missing="drop",
    factor_levels={"treatment": ("control", "treated")},
)

print(audited.audit.input_rows)
print(audited.audit.analysis_rows)
for record in audited.audit.records:
    if record.action != "retained":
        print(record.reason_code, record.source_index)
```

The audit has one disposition per input row. It records factor levels,
contrast coding, transformations, source and analysis fingerprints, and the
mapping from source positions to likelihood rows.

## Repeated-measures invariants

Use:

- `find_duplicate_keys` to identify duplicate combinations such as
  `(subject, visit)`;
- `validate_monotonic_time` to check within-group time direction;
- `stable_sort` to order rows deterministically while retaining ties.

MMRM covariance does not infer adjacency from arbitrary input row order. The
scientific visit axis is compiled and archived separately.

## Formula anatomy

```text
response ~ fixed terms + (random terms | group)
```

Examples:

| Formula | Meaning |
|---|---|
| `y ~ x` | Fixed intercept and slope for `x` |
| `y ~ 0 + x` | Slope without fixed intercept |
| `y ~ a + b + a:b` | Main effects and interaction |
| `y ~ a * b` | `a + b + a:b` |
| `y ~ a / b` | Nested shorthand expanded deterministically |
| `y ~ x + (1 | subject)` | Random intercept |
| `y ~ x + (1 + x | subject)` | Correlated random intercept and slope |
| `y ~ x + (1 + x || subject)` | Independent random intercept and slope |
| `y ~ x + (1 | site) + (1 | subject)` | Two random grouping blocks |

`parse_formula` creates a `FormulaSpec` without data. `compile_formula` (alias
`dry_run`) creates audited fixed/random design matrices from data.

```python
from pymixef.formula import explain_formula, parse_formula

spec = parse_formula("y ~ treatment * time + (1 + time || subject)")
print(spec)
print(explain_formula(spec, data))
```

## Safe transformations

The formula evaluator accepts an allowlist:

| Transform | Purpose |
|---|---|
| `I(expression)` | protect arithmetic as one formula term |
| `C(x)` | categorical encoding |
| `center(x)` | subtract mean |
| `scale(x)`, `standardize(x)` | center and divide by standard deviation |
| `poly(x, degree)` | deterministic polynomial basis |
| `abs`, `cos`, `exp`, `log`, `log1p`, `sin`, `sqrt` | elementwise transforms |

Arbitrary calls, attribute access, imports, and unregistered names are rejected.
This is a statistical expression language, not Python evaluation.

```python
model = pymixef.Model.from_formula(
    "response ~ treatment + center(time) + I(time ** 2) + (1 | subject)"
)
```

## Factor levels and reference categories

Factor encoding is deterministic. Ordered categorical metadata is honored when
available; otherwise levels follow the compiler’s documented stable ordering.
Inspect:

```python
plan = model.compile(data, engine="lmm", method="reml")
print(plan.matrices.factor_levels)
print(plan.matrices.fixed_names)
```

For confirmatory work, make desired levels explicit in the input type or audit
contract instead of relying on incidental data appearance order.

## Rank and identifiability checks

Compilation reports fixed-design shape and rank. A full-rank matrix is necessary
for the declared fixed effects but does not establish practical identifiability
of covariance parameters, robustness to sparse cells, or scientific relevance.
Review group counts, within-group replication, factor cells, visit coverage, and
boundary diagnostics before fitting.

## API map

- `pymixef.data`: adapters, schemas, missingness, audits, repeated-key helpers.
- `pymixef.formula`: parser, design compiler, explanations, safe transforms.
- `Model.from_formula`: backend-neutral model declaration.
- `Model.explain` and `ExecutionPlan.explain`: pre-fit inspection.

See the generated [data API](../api/generated/pymixef.data.rst) and
[formula API](../api/generated/pymixef.formula.rst).

