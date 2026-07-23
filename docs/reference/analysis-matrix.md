# Analysis matrix

This matrix distinguishes what the 0.1 reference paths calculate from what the
broader object catalog can represent.

## Fitted statistical models

| Analysis | Response | Dependence | Engine/method | Principal outputs | Current boundary |
|---|---|---|---|---|---|
| Gaussian LMM | continuous | one or more Gaussian random blocks; supported residual structures | `lmm`, ML/REML | fixed effects/covariance, natural-scale variance components, modes, conditional/population fitted values, residuals, simulation | dense experimental reference path for small/moderate problems; not compiled sparse scale |
| Bernoulli GLMM | binary | Gaussian random blocks | `glmm`, first-order Laplace | link-scale coefficients, conditional ORs after transformation, conditional modes, fitted probabilities, convergence/approximation metadata | canonical logit reference path; no AGHQ |
| Binomial GLMM | successes/trials | Gaussian random blocks | `glmm`, first-order Laplace | conditional log-odds effects, modes, fitted means | supported parameterization must validate |
| Poisson GLMM | counts | Gaussian random blocks | `glmm`, first-order Laplace | conditional log-rate effects, modes, fitted counts | exposure/offset meaning is analyst responsibility |
| NB2 GLMM | overdispersed counts | Gaussian random blocks | `glmm`, first-order Laplace | conditional mean and NB2 dispersion path | NB1 and other catalog families are not fit by this engine |
| MMRM | continuous repeated visits | dense within-subject residual covariance, between-subject independence | `mmrm`, ML/REML | fixed effects/covariance, visit covariance, linear inference, labeled DF, exclusions/visit order | no simultaneous formula random effects; exact KR is not claimed |

## MMRM covariance choices

| Structure | Heterogeneous variance | Lag/distance behavior | Ordering requirement |
|---|---:|---|---|
| Diagonal | optional by declaration | no off-diagonal correlation | visit identity |
| Unstructured | yes | every covariance free | explicit visit axis |
| Compound symmetry | no | common correlation | visit identity |
| AR(1) | no | geometric decay by ordered lag | ordered visits |
| Heterogeneous AR(1) | yes | geometric decay by ordered lag | ordered visits |
| Toeplitz | no | separate correlation per lag | ordered visits |
| Heterogeneous Toeplitz | yes | separate correlation per lag | ordered visits |
| Ante-dependence | yes/flexible | sequential conditional dependence | ordered visits |
| Spatial power | structure-defined | numeric distance | numeric visit times/distances |
| Known covariance | supplied | supplied | axis must match matrix |

## Pharmacometric calculations

| Task | Entry points | Output | Estimation? |
|---|---|---|---|
| Canonical event preparation | `canonicalize_events`, `EventTable` expansion methods | immutable event table + row/action audit | no |
| One-compartment PK | bolus, infusion, oral helpers / `OneCompartmentPK` | structural concentration | no |
| Two-compartment PK | bolus, infusion, oral helpers / `TwoCompartmentPK` | structural concentration and derived rates | no |
| Event-aware ODE | `simulate_ode`, `simulate_subjects` | state trajectories, observations, metadata, sensitivities | no |
| Residual-error likelihood | additive/proportional/power/combined/lognormal; censoring helpers | variance, draws, log likelihood | evaluates supplied parameters |
| Typed model contract | `@model`, declarations, `CompiledModel` | validated model + ModelIR | no |
| Conditional ETA mode | `ConditionalObjective`, `find_conditional_mode` | mode, objective components, gradient/Hessian/covariance evidence | subject-level primitive |
| Population Laplace aggregation | `laplace_population_objective` | subject contributions/modes + objective | primitive, not integrated outer fit |
| FOCEI | `fit_focei` | explicit `UnsupportedEstimatorError` | unavailable |
| SAEM | `experimental_saem` / `saem` | callback-driven research-kernel result | experimental primitive, not integrated estimator |

## Post-fit analysis

| Need | API | Key contract |
|---|---|---|
| Human summary | `FitResult.summary()` | readable view; structured fields remain authoritative |
| Prediction | `FitResult.prediction(mode=...)` | mode is explicit |
| Residual diagnostics | `residual_diagnostics`, `residual_table` | machine-readable row-aligned table |
| Named diagnostics | `FitResult.diagnostic` | returns `DiagnosticTable` |
| Simulation | `FitResult.simulate` | seed and included uncertainty sources recorded |
| VPC | `FitResult.vpc`, `vpc_table` | bin/quantile data, not a pass/fail certificate |
| Bootstrap | `bootstrap` | row/cluster resampling, checkpoint/resume, failure accounting |
| Linear inference | MMRM `linear_inference` | explicit coefficient mapping and DF label |
| External comparison | `compare` | parameter mapping and convention matching |
| Report | `render_report` | Markdown/HTML; optional PDF/Word dependencies |
| Archive | `FitResult.save/load` | inspectable JSON + integrity sidecar |
| Validation bundle | create/verify helpers | evidence package; data excluded by default |

## Representable but not automatically fitted

The family catalog includes Student-t, lognormal, gamma, inverse Gaussian, beta,
Tweedie, NB1, generalized Poisson, COM-Poisson, ordinal, multinomial,
zero-inflated, hurdle, truncated, censored, and survival objects. ModelIR can
represent priors, distributional predictors, and richer typed nodes.
Representation supports calculation, exchange, or future backend contracts; it
does not imply that a current formula backend can estimate the model.

## Unavailable integrated paths

The 0.1 release does not provide:

- adaptive Gauss–Hermite quadrature;
- a production FOCEI or integrated SAEM population estimator;
- exact Kenward–Roger inference;
- a compiled sparse million-row LMM backend;
- finite-mixture estimation;
- advanced MCMC/samplers;
- joint longitudinal-event estimation;
- automatic missing-data sensitivity recipes;
- universal external-format translation or regulatory qualification.

Unsupported requests are validated and refused rather than silently replaced.
For machine-readable release state, run `pymixef capabilities --json`.

