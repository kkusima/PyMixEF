# Developer guide

## Static typing

The 0.1 release ships useful inline types and a strict staged gate for 26
foundation/public-contract modules:

```bash
make typecheck-core
```

`make typecheck-all` intentionally exposes the remaining annotation and numeric
array-typing debt in the experimental family and estimator implementations.
Passing the core gate is not represented as whole-package strict typing.

Read the [architecture guide](development/architecture.md) before extending
PyMixEF.

New families provide support, normalized `log_prob`, a CDF where censoring needs
it, controlled random generation, parameter names, and derivative behavior. New
covariance kernels must validate inputs, return a positive-definite matrix by
construction (or document semidefinite behavior), simulate, and expose the
unconstrained-to-natural mapping.

New engines implement the backend Protocol and common payload, register static
compatibility, return a structured convergence object, and declare a
reproducibility class. The current reusable fit-contract suite covers every
built-in backend and fails if a newly registered built-in has no case. A new
backend must pass {py:func}`pymixef.backends.base.validate_payload` and those
behavior checks. `ARCH-003` nevertheless remains open because the Backend
Protocol does not yet expose the blueprint's separate objective, gradient,
optional Hessian-vector product, and simulation contracts.

Numerical changes require independent derivative/reference tests, pathology
tests, a change-impact category, and targeted benchmark reruns. Public model
semantics and IR changes require an RFC and schema migration.
