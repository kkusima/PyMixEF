# The core workflow

PyMixEF separates scientific declaration from numerical execution. Each stage
creates an object that can be inspected, tested, and archived.

```text
data + model declaration
        ↓
validate compatibility
        ↓
compile immutable execution plan + ModelIR
        ↓
fit with a named engine and method
        ↓
check structured convergence
        ↓
infer, predict, diagnose, simulate
        ↓
archive result + provenance + validation evidence
```

## 1. Adapt and audit data

`adapt_data` normalizes supported columnar containers. `audit_data` applies
required-column and missingness policy while preserving source-row identity.
Duplicate-key and monotonic-time helpers address common repeated-measures
invariants.

```python
from pymixef.data import audit_data

audited = audit_data(
    data,
    response="response",
    covariates=("time", "subject"),
    missing="drop",
)
print(audited.audit.to_dict())
```

Review which rows enter the likelihood. A deterministic exclusion is auditable;
it is not automatically scientifically appropriate.

## 2. Declare once

Formula route:

```python
model = pymixef.Model.from_formula(
    "response ~ treatment * time + (1 | subject)"
)
```

Builder route:

```python
model = pymixef.Model(
    response=pymixef.Response("response"),
    fixed=pymixef.Fixed("treatment * time"),
    random=[pymixef.Random("1", group="subject")],
)
```

Pharmacometric DSL route:

```python
from pymixef.pharmacometrics import Param, State, model

@model
def structural_model():
    clearance = Param.positive("clearance", init=5.0)
    central = State("central", initial=0.0)
    return clearance, central
```

The authoring interfaces converge on a typed, versioned intermediate
representation rather than sending uninspected user syntax directly to an
optimizer.

## 3. Validate support explicitly

```python
report = model.validate(data, engine="glmm", method="laplace")
if not report.valid:
    for finding in report.findings:
        print(finding.to_dict())
    report.raise_for_errors()
```

Validation checks the requested combination. For example, requesting AGHQ does
not silently substitute Laplace. Compatibility failures carry stable codes and
actionable messages.

## 4. Compile without fitting

```python
plan = model.compile(data, engine="lmm", method="reml")
print(plan.explain())
print(plan.model_ir.semantic_hash)
```

Compilation fixes:

- analysis rows and source-row reconciliation;
- factor levels and reference categories;
- fixed- and random-design matrices;
- visit ordering and covariance axes where relevant;
- family/link and engine/method;
- numerical options;
- a deterministic `ModelIR`.

That makes “what was fitted?” answerable before optimization begins.

## 5. Fit with a named calculation path

```python
result = plan.fit()
```

The result records objective, log likelihood, parameters, warnings, convergence,
engine/method, model and data fingerprints, environment, and backend-specific
diagnostic inputs. Calculation paths are explicitly labeled as reference or
experimental where applicable.

## 6. Gate interpretation on numerical evidence

```python
convergence = result.convergence
print(convergence.to_dict())

if not convergence.trustworthy:
    raise RuntimeError("Fit requires numerical review")
```

Review optimizer termination, gradient scale, Hessian diagnostics when
available, boundary records, conditional-mode failures for GLMMs, rescue steps,
and stable warning codes. `success` is a convenience property; `trustworthy`
encodes the stronger package-level gate.

## 7. Match output to the estimand

Fixed effects, random-effect modes, conditional predictions, population
predictions, contrasts, residuals, simulations, and VPC tables answer different
questions. Preserve labels and transformation scales when reporting them.

Use `compare` for structured cross-result comparison, `bootstrap` for
restartable row- or cluster-resampling, and MMRM `linear_inference` for declared
linear functions of fixed effects.

## 8. Diagnose with data, not only pictures

Diagnostic methods return `DiagnosticTable` objects. Plots should be derived
from these auditable values:

```python
residuals = result.residual_diagnostics()
simulated = result.simulate(n_replicates=500, seed=2026)
vpc = result.vpc(simulations=500, seed=2026)
```

A centered residual plot or an observed curve inside a simulation envelope is
evidence to inspect, not a universal acceptance test.

## 9. Preserve the evidence

```python
result.save("analysis-result.json")

bundle = pymixef.create_validation_bundle(
    result,
    "validation-bundle",
    include_data=False,
)
verification = pymixef.verify_validation_bundle(bundle)
assert verification["valid"]
```

The result archive and validation bundle support reproducibility and
traceability. Hashes detect content change but do not establish authorship,
approval, model validity, or regulatory qualification.

## 10. Revisit capability evidence

Methods evolve independently. Query capabilities in code:

```python
for capability in pymixef.iter_capabilities():
    print(capability.name, capability.maturity, capability.evidence)
```

or from the CLI:

```bash
pymixef capabilities
pymixef capabilities --json
```

The registry is the authoritative disclosure of maturity and open gates for
this release.
