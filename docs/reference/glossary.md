# Glossary

```{glossary}
Analysis row
  An input row retained in the likelihood after the explicit data and
  missingness contract is applied.

Approximation
  A named numerical/statistical replacement for an exact calculation, such as
  first-order Laplace integration around a conditional mode.

Boundary estimate
  A fitted parameter at or numerically near the edge of its admissible space,
  such as a variance near zero or correlation near ±1.

Canonical event
  A normalized pharmacometric record with explicit event type, deterministic
  same-time priority, amount semantics, row identity, and source provenance.

Conditional mode
  The mode of a group/subject random effect given observations and population
  parameters; often called an empirical Bayes mode. It is estimated and shrunk,
  not directly observed.

Conditional prediction
  Prediction that includes fitted random effects for observed groups.

Covariance axis
  The ordered visits or times to which a structured repeated-measures covariance
  matrix refers.

Data audit
  One disposition per source row plus factor, transformation, and fingerprint
  information connecting input data to analysis data.

Degrees of freedom (DF)
  A named reference distribution/calculation used for finite-sample inference.
  PyMixEF preserves labels such as residual or Satterthwaite delta-method.

Engine
  A numerical backend that consumes a compatible compiled model/data payload,
  such as `lmm`, `glmm`, or `mmrm`.

Estimand
  The precisely defined quantity an analysis aims to estimate, including
  population, endpoint, treatment condition, handling of intercurrent events,
  and summary measure where relevant.

Family
  A conditional response distribution and its probability contract.

Fixed effect
  A population-level coefficient in the linear predictor.

FOCEI
  First-order conditional estimation with interaction. The integrated
  production estimator is not available in PyMixEF 0.1.

GLMM
  Generalized linear mixed model: a non-Gaussian response model with a link
  function and random effects.

Integrity hash
  A cryptographic digest used to detect content change. It is not a digital
  signature or scientific-validity assessment.

Laplace approximation
  Approximation of an integral from local curvature around a mode. The current
  GLMM path uses a first-order Laplace approximation.

Link
  A transformation connecting a conditional response mean to the linear
  predictor.

LMM
  Linear mixed model: a Gaussian response model with fixed and random effects.

Manifest
  Structured provenance describing software, environment, engine/method,
  options, warnings, timing, and model/data fingerprints.

Marginal prediction
  A prediction integrated over random effects. It is not generally identical to
  setting the random effect to zero in a nonlinear-link model.

MMRM
  Mixed model for repeated measures: in this documentation, a Gaussian
  longitudinal model with explicit within-subject residual covariance and
  between-subject independence.

ModelIR
  PyMixEF’s immutable, typed, versioned, backend-neutral model intermediate
  representation.

Population prediction
  Fixed-effect or typical-group prediction with random effects at their
  reference value. In a nonlinear GLMM this is conditional-at-zero, not
  necessarily random-effect-integrated marginal prediction.

Random effect
  A latent group-specific deviation drawn from a declared covariance model.

REML
  Restricted maximum likelihood, which integrates/adjusts for fixed effects
  under a stated constant convention when estimating Gaussian covariance
  parameters.

Reproducibility class
  Manifest label stating whether a computation is deterministic, deterministic
  within numerical tolerance, or stochastic with a recorded seed/Monte Carlo
  error.

Residual
  Observed minus fitted response under a named prediction/variance convention.

SAEM
  Stochastic approximation expectation maximization. PyMixEF 0.1 exposes a
  callback-driven experimental research kernel, not an integrated population
  estimator.

Semantic hash
  Deterministic hash of model meaning after canonical serialization, excluding
  nonsemantic formatting.

Source row
  Original input record identified by source position/index and a stable row ID.

Trustworthy convergence
  The package’s combined interpretation gate over optimizer termination,
  gradients, curvature/boundaries, inner modes, and warning state.

Validation bundle
  Deterministic collection of result, manifest, traceability, hashes, and
  optionally authorized data that supports a context-specific validation
  process.

Visual predictive check (VPC)
  Comparison of observed summaries with distributions of analogous summaries
  from seeded model simulations, returned by PyMixEF as auditable table data.
```

