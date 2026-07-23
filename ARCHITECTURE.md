# PyMixEF implementation architecture

PyMixEF follows the blueprint's central rule: a scientific model is represented
once in a versioned, immutable intermediate representation (IR), while estimation
engines are replaceable consumers.

The public import name and distribution are both `pymixef`; the human-facing name
is **PyMixEF**.

## Layer boundaries

- `ir.py`, `formula.py`, `data.py`, `transforms.py`, and `covariance.py` define
  semantics and compilation. They never optimize a model.
- `backends/` consumes compiled matrices or model nodes and returns a
  backend-neutral payload.
- `model.py` owns public builders, validation, engine compatibility checks, and
  dispatch.
- `results.py`, `convergence.py`, `diagnostics.py`, and `provenance.py` own stable
  result contracts.
- `pharmacometrics/` owns canonical event records, typed model declarations,
  ODE integration, closed-form PK, and population estimation helpers.
- `interoperability/` always returns a compatibility report and refuses
  unsupported translations.

## Shared backend payload contract

Backends return a mapping with these keys:

- `parameters`: natural-scale named estimates.
- `unconstrained_parameters`: optimizer-scale estimates.
- `parameter_covariance`: optional square array.
- `fitted_values`, `residuals`, `random_effects`: numeric outputs.
- `objective`, `log_likelihood`, `method`, `engine`: calculation metadata.
- `convergence`: mapping accepted by `ConvergenceReport.from_dict`.
- `diagnostic_data`: mapping of tidy column-oriented tables.
- `extra`: engine-specific serializable values.

Backends raise typed `PyMixEFError` subclasses for invalid or incompatible
models. They must never silently select a scientifically different estimator.

## Capability policy

Implemented paths declare their maturity and reproducibility class. Planned or
partially specified algorithms are represented in the capability registry but
are not represented as complete. Estimator entry points that exist for gated
methods fail before optimization with a stable unsupported code and, where
applicable, list the available lower-level primitives.

The initial release implements foundation contracts and selected dense reference
paths toward the Classical Core. It has not passed either blueprint stage gate.
The reusable fit-payload suite covers every built-in backend, including
deterministic repeat fitting, row alignment, and input immutability. It is not
the full `ARCH-003` protocol: separate objective, gradient, optional
Hessian-vector product, and simulation contracts remain open. Later-stage
authoring and interchange semantics exist so those engines can be added
without changing public model meaning; representability does not mean
estimability.
