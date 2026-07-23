# Typed pharmacometric model authoring

The pharmacometric DSL executes ordinary Python declarations to build typed
symbolic objects. It does not parse arbitrary model text. The output is a
validated {py:class}`pymixef.pharmacometrics.dsl.CompiledModel` and then the
same versioned {py:class}`pymixef.ir.ModelIR` used by the rest of PyMixEF.

## Declare a model

```python
from pymixef.pharmacometrics import (
    Dose,
    Eta,
    Param,
    State,
    additive,
    covariate,
    d,
    exp,
    model,
    observe,
)

@model(name="one_compartment_population_pk")
def population_pk():
    tvcl = Param.positive("tvcl", init=5.0, unit="L/h")
    volume = Param.positive("volume", init=25.0, unit="L")
    sigma = Param.positive("sigma", init=0.2, unit="mg/L")

    eta_cl = Eta.independent("eta_cl", block="omega_cl", level="subject")
    weight = covariate("weight", unit="kg", reference=70.0)
    central = State("central", unit="mg", initial=0.0)

    clearance = tvcl * (weight / 70.0) ** 0.75 * exp(eta_cl)
    equation = d(central, -(clearance / volume) * central)
    dose = Dose.into(central, amount="AMT", rate="RATE", duration="DUR")
    observation = observe(
        "DV",
        mean=central / volume,
        error=additive(sigma),
    )
    return equation, dose, observation

compiled = population_pk()
print(compiled.explain())
```

The {py:func}`pymixef.pharmacometrics.dsl.model` decorator returns a
{py:class}`pymixef.pharmacometrics.dsl.ModelDefinition`. Calling it executes the
declaration, collects returned components, validates them, and creates a
{py:class}`pymixef.pharmacometrics.dsl.CompiledModel`.

The example declares the symbolic equations

$$
CL_i=TVCL\left(\frac{WT_i}{70}\right)^{0.75}e^{\eta_{CL,i}},
\qquad
\frac{dA_i}{dt}=-\frac{CL_i}{V}A_i,
\qquad
f_i=\frac{A_i}{V}.
$$

## Symbolic building blocks

| Object/helper | Role |
|---|---|
| {py:meth}`~pymixef.pharmacometrics.dsl.Param.real`, {py:meth}`~pymixef.pharmacometrics.dsl.Param.positive`, {py:meth}`~pymixef.pharmacometrics.dsl.Param.bounded` | declared fixed/model parameters and constraints |
| {py:meth}`~pymixef.pharmacometrics.dsl.Eta.independent`, {py:meth}`~pymixef.pharmacometrics.dsl.Eta.correlated` | random effects, covariance block, and hierarchy level |
| {py:func}`~pymixef.pharmacometrics.dsl.symbol`, {py:func}`~pymixef.pharmacometrics.dsl.covariate` | predictors with role, unit, and reference |
| {py:class}`~pymixef.pharmacometrics.dsl.State` | dynamic state and initial value |
| {py:func}`~pymixef.pharmacometrics.dsl.d` / {py:func}`~pymixef.pharmacometrics.dsl.derivative` | differential equation for a state |
| {py:meth}`~pymixef.pharmacometrics.dsl.Dose.into` | event fields mapped into a state |
| {py:func}`~pymixef.pharmacometrics.dsl.observe` | endpoint, structural mean, error, and optional censoring |
| DSL {py:func}`~pymixef.pharmacometrics.dsl.exp`, {py:func}`~pymixef.pharmacometrics.dsl.log`, {py:func}`~pymixef.pharmacometrics.dsl.log1p`, {py:func}`~pymixef.pharmacometrics.dsl.sqrt` | typed symbolic transforms |

`Expr.format()` supports review, `Expr.to_dict()` supports serialization, and
`Expr.evaluate()` evaluates against an explicit symbol environment.

## Constraints and transforms

Parameter declarations preserve support:

- real → identity transform;
- positive → log transform;
- bounded → bounded transform with explicit lower/upper values.

Constraints become {py:class}`pymixef.ir.ParameterIR` and
{py:class}`pymixef.ir.TransformIR` nodes. Units are metadata and
are not automatically converted.

## Random-effect blocks

An {py:class}`pymixef.pharmacometrics.dsl.Eta` names the random effect, block,
covariance mode (`diagonal` or
`correlated`), and level such as subject. The DSL can represent richer blocks
than the integrated estimator currently supports; check validation compatibility
rather than assuming representability implies estimability.

## Validate capability

```python
validation = compiled.validate()
print(validation.valid)
print(validation.dimensions)
print(validation.estimator_compatibility)

for message in validation.messages:
    print(message.to_dict())
```

Compatibility entries currently distinguish simulation, conditional-mode,
FOCEI, and SAEM support. A syntactically valid model can still be incompatible
with an estimator.

## Compile to ModelIR

```python
ir = compiled.to_ir()
print(ir.schema_version)
print(ir.semantic_hash)
print(ir.to_json(indent=2))
```

The IR contains typed parameter, random-effect, predictor, state-equation,
event, likelihood, transform, and output nodes with explicit dependencies.
Round-trip with {py:meth}`pymixef.ir.ModelIR.from_json`; semantic equality and hash preservation
are testable.

Node counts and a valid dependency graph describe the model contract. They do
not establish identifiability, estimator readiness, biological plausibility, or
adequate data.

## Programmatic construction

{py:func}`pymixef.pharmacometrics.dsl.compiled_model` constructs a
{py:class}`pymixef.pharmacometrics.dsl.CompiledModel` directly from component
sequences. {py:meth}`pymixef.pharmacometrics.dsl.ModelDefinition.compile`,
{py:meth}`~pymixef.pharmacometrics.dsl.ModelDefinition.validate`,
{py:meth}`~pymixef.pharmacometrics.dsl.ModelDefinition.explain`,
{py:meth}`~pymixef.pharmacometrics.dsl.ModelDefinition.to_dict`,
{py:meth}`~pymixef.pharmacometrics.dsl.ModelDefinition.to_ir`, and
{py:attr}`~pymixef.pharmacometrics.dsl.ModelDefinition.declaration_signature`
expose each stage without requiring an implicit fit.

## API and tutorial

- {py:mod}`pymixef.pharmacometrics.dsl`
- {py:mod}`pymixef.ir`
- [Executed DSL and ModelIR tutorial](../tutorials/09-pharmacometrics-dsl-and-model-ir.md)
