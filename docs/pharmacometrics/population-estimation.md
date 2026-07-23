# Population-estimation primitives

This module provides transparent building blocks for research and validation of
population-estimation algorithms. It does not currently provide an integrated
production FOCEI or SAEM workflow.

## Parameter/random-effect mapping

{py:func}`pymixef.pharmacometrics.estimation.apply_random_effects` combines
typical values and ETAs using named relationships.
{py:func}`pymixef.pharmacometrics.estimation.omega_from_standard_deviations`
creates a covariance matrix from SDs and an optional correlation matrix.
{py:func}`pymixef.pharmacometrics.estimation.eta_shrinkage` summarizes empirical
ETA variance relative to the declared omega.

The common exponential mapping and covariance construction are

$$
\theta_{ik}=\theta_{\mathrm{typ},k}\exp(\eta_{ik}),
\qquad
\eta_i\sim N(0,\Omega),
\qquad
\Omega=D_\omega R D_\omega,
$$

where $D_\omega=\operatorname{diag}(\omega_1,\ldots,\omega_q)$ and $R$ is the
declared ETA correlation matrix.

```python
from pymixef.pharmacometrics import (
    apply_random_effects,
    omega_from_standard_deviations,
)

omega = omega_from_standard_deviations([0.3, 0.2])
individual = apply_random_effects(
    {"CL": 5.0, "V": 20.0},
    {"CL": 0.1, "V": -0.05},
    relationships={"CL": "exponential", "V": "exponential"},
)
```

## Conditional objective and mode

{py:class}`pymixef.pharmacometrics.estimation.ConditionalObjective` combines observations, a caller-provided prediction
function of ETA, omega, an `ObservationError`, optional error parameters, and
censoring information.

For subject $i$, the conditional mode minimizes

$$
Q_i(\eta_i;\theta)
=-\log p(y_i\mid\eta_i,\theta)
+\frac12\eta_i^\mathsf{T}\Omega^{-1}\eta_i
+\frac12\log|\Omega|
+\frac{q}{2}\log(2\pi).
$$

{py:func}`pymixef.pharmacometrics.estimation.find_conditional_mode` returns:

- ETA mode;
- total, observation, and random-effect objective components;
- gradient and Hessian;
- covariance when available;
- convergence state/message and counts;
- gradient norm, Hessian positive-definiteness, and warning codes.

```python
mode = find_conditional_mode(
    objective,
    initial_eta=[0.0, 0.0],
    method="BFGS",
    tolerance=1e-8,
    max_iterations=500,
    require_success=True,
)
```

{py:func}`pymixef.pharmacometrics.estimation.conditional_mode_objective` exposes
the scalar calculation directly.
{py:func}`pymixef.pharmacometrics.estimation.finite_difference_gradient` and
{py:func}`pymixef.pharmacometrics.estimation.finite_difference_hessian` support transparent
numerical checks.

## Laplace population objective

{py:func}`pymixef.pharmacometrics.estimation.laplace_population_objective`
evaluates a sequence of subject objectives,
finds their conditional modes, and returns total objective, subject
contributions, modes, and warning codes. `require_modes=True` prevents silent
continuation after a failed inner mode.

This is a calculation primitive. An integrated population optimizer must also
manage parameter transforms, outer optimization, covariance updates,
identifiability, error recovery, and validation evidence.

## FOCEI boundary

{py:func}`pymixef.pharmacometrics.estimation.fit_focei` deliberately raises
{py:class}`pymixef.pharmacometrics.estimation.UnsupportedEstimatorError`. The explicit
refusal prevents a partial Laplace calculation from being presented as a
production FOCEI implementation.

## Experimental SAEM kernel

{py:class}`pymixef.pharmacometrics.estimation.SAEMProblem` is callback-driven:
the caller supplies the joint log density, sufficient-statistic calculation,
and M-step. {py:class}`pymixef.pharmacometrics.estimation.SAEMControl` defines iterations,
burn-in, stochastic-approximation schedule, MCMC steps, proposal scale, seed,
and latent-trace retention.

{py:func}`pymixef.pharmacometrics.estimation.experimental_saem` (alias
{py:func}`pymixef.pharmacometrics.estimation.saem`) returns parameter/latent state, sufficient
statistics, traces, step sizes, acceptance accounting, seed, burn-in, warning
codes, and an explicit `experimental=True` marker.

It is a research kernel, not an integrated NONMEM-class population estimator.
The caller owns proposal suitability, complete-data model correctness,
convergence diagnosis, Monte Carlo error analysis, and external validation.

## Failure types

- {py:class}`pymixef.pharmacometrics.estimation.EstimationError`: shared estimation contract failure.
- {py:class}`pymixef.pharmacometrics.estimation.ConditionalModeError`: inner ETA-mode failure.
- {py:class}`pymixef.pharmacometrics.estimation.UnsupportedEstimatorError`: requested estimator is intentionally unavailable.
- {py:class}`pymixef.pharmacometrics.estimation.SAEMError`: invalid problem/control or stochastic-kernel failure.

## API

See
{py:mod}`pymixef.pharmacometrics.estimation`
for every primitive and result field.
