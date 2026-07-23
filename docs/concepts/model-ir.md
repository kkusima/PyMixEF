# Model IR and compilation

Every PyMixEF authoring surface compiles to a
{py:class}`pymixef.ir.ModelIR`. The IR stores mathematical meaning rather than
the source syntax:

- response and fixed predictors;
- random-effect terms, grouping, and covariance;
- likelihood family, links, and distributional predictors;
- residual covariance;
- natural and unconstrained parameter transforms;
- units, priors, states, events, and requested outputs;
- source locations and schema version.

IR objects are immutable and serialize to canonical JSON. Unknown future schema
major versions are rejected. Migrations must be explicit and tested.

The formula parser ({py:meth}`pymixef.formula.FormulaSpec.to_ir`), structured
public builder ({py:meth}`pymixef.model.Model.to_ir`), and pharmacometrics
declaration layer ({py:meth}`pymixef.pharmacometrics.dsl.CompiledModel.to_ir`)
all produce this same schema. Pharmacometric
parameters, random-effect blocks, covariates, ODEs, dose mappings, residual
error models, and observation outputs remain typed nodes rather than opaque
backend metadata. Custom expression or error-model operations without a
defined IR mapping are rejected explicitly.

Public {py:class}`pymixef.model.Model` declarations become
{py:class}`pymixef.ir.PriorIR` nodes. A prior
can be a distribution name, a mapping containing `distribution` plus inline
or nested `parameters`, or an already constructed `PriorIR`; it is never
stored only as untyped metadata.

{py:meth}`pymixef.model.Model.validate` checks syntax and static estimator
compatibility. {py:meth}`pymixef.model.Model.compile` additionally builds
matrices, audits rows, fixes factor-level
order, checks ranks, and creates an execution plan. Neither method optimizes.

Formula expressions use a safe grammar. Arbitrary Python or R code is never
evaluated. Transform a column explicitly when a migration report says an R
function call is unsupported.

## Typed node inventory

| Node | Meaning |
|---|---|
| {py:class}`~pymixef.ir.ParameterIR` | named value/support/bounds/unit and optimizer transform |
| {py:class}`~pymixef.ir.FixedEffectIR` | population predictor term and coefficient |
| {py:class}`~pymixef.ir.RandomEffectIR` | grouped latent effect, terms, level, and covariance |
| {py:class}`~pymixef.ir.PredictorIR` | covariate/symbol role, reference, and unit |
| {py:class}`~pymixef.ir.LikelihoodIR` | endpoint, family, link, prediction, and error metadata |
| {py:class}`~pymixef.ir.CovarianceIR` | structured covariance name, axis, grouping, and parameters |
| {py:class}`~pymixef.ir.TransformIR` | natural ↔ unconstrained parameter mapping |
| {py:class}`~pymixef.ir.PriorIR` | prior target, distribution, and parameters |
| {py:class}`~pymixef.ir.StateEquationIR` | dynamic state, derivative expression, and dependencies |
| {py:class}`~pymixef.ir.EventIR` | dose/event mapping into model state |
| {py:class}`~pymixef.ir.OutputIR` | named model output and dependencies |

Every node inherits the immutable {py:class}`pymixef.ir.IRNode` contract and has
a JSON-compatible {py:meth}`~pymixef.ir.IRNode.to_dict`. Node dependencies make
the model graph inspectable without executing an estimator.

## Serialize and verify identity

```python
ir = model.to_ir()

document = ir.to_json(indent=2)
loaded = pymixef.ModelIR.from_json(document)

assert loaded == ir
assert loaded.semantic_hash == ir.semantic_hash
```

{py:meth}`ModelIR.to_json <pymixef.ir.ModelIR.to_json>` without indentation
returns canonical compact JSON:

- mapping keys are deterministically ordered;
- nonfinite numeric values are refused;
- tuples/mappings are frozen inside the object;
- the SHA-256 `semantic_hash` is computed from canonical content.

`semantic_hash` and its `hash` alias are properties, not methods.

## Validation and versions

The current schema version is exposed as `MODEL_IR_SCHEMA_VERSION` (also
`IR_SCHEMA_VERSION`). `SUPPORTED_IR_SCHEMA_VERSIONS` lists accepted historical
versions.

{py:meth}`pymixef.ir.ModelIR.from_dict` and
{py:meth}`pymixef.ir.ModelIR.from_json` validate the full document. Unknown fields,
wrong node types, duplicate parameter names, invalid references, and unsupported
future major versions raise {py:class}`pymixef.errors.IRValidationError` or
{py:class}`pymixef.errors.IRVersionError` rather than
being discarded.

Set `migrate=False` when a workflow must reject any older document:

```python
ir = pymixef.ModelIR.from_json(payload, migrate=False)
```

## Explicit migrations

{py:func}`pymixef.ir.migrate_ir` walks registered forward edges.
{py:func}`pymixef.ir.register_ir_migration` adds a deliberate
migration. A migration is code, versioned behavior, and should have fixtures
that demonstrate preservation or explicitly document semantic change.

Unknown migration paths are refused; there is no “best effort” dropping of
future fields.

## Compare model meaning

```python
from pymixef import diff_models

difference = diff_models(before_ir, after_ir)
print(difference.categories)
print(difference.to_json(indent=2))
assert not difference.equal
```

{py:class}`pymixef.ir.ModelDiff` contains deterministic
{py:class}`pymixef.ir.DiffEntry` records with path, before/after value, and
change category. {py:meth}`pymixef.ir.ModelIR.diff` is the object-oriented form.
Use the diff to drive change review; deciding whether a change is scientifically
material remains a context-specific judgment.

## What the hash does—and does not—establish

The semantic hash is strong evidence that two canonical IR documents are
identical. It does not prove:

- that data, numerical options, or software environment are identical;
- that a model is identifiable or scientifically appropriate;
- that two external tools use the same likelihood constants;
- authorship, approval, or regulated-system validation.

The {py:class}`pymixef.provenance.RunManifest` combines the model hash with data
fingerprint, environment, engine/method, controls, and warnings.

## API and example

- {py:mod}`pymixef.ir` documents every node, property,
  migration, and diff type.
- [Tutorial 09](../tutorials/09-pharmacometrics-dsl-and-model-ir.md) visualizes a
  compiled dependency graph, node inventory, and incidence matrix, then verifies
  JSON/hash preservation.
