# Linear mixed models

A Gaussian linear mixed model (LMM) combines population-level fixed effects
with latent group-specific deviations.

$$
y=X\beta+Zb+\epsilon,\qquad
b\sim N(0,G),\qquad
\epsilon\sim N(0,R).
$$

The marginal covariance is $V=ZGZ^\mathsf{T}+R$. PyMixEF profiles
$\beta$, evaluates the normalized Gaussian likelihood with stable Cholesky
solves, and optimizes unconstrained covariance parameters that map to
positive-definite natural-scale matrices.

## When to use an LMM

An LMM is a useful starting point when:

- the conditional response is reasonably modeled as continuous Gaussian;
- observations share groups such as subject, batch, center, lot, device, or
  laboratory;
- random intercepts/slopes are a scientifically meaningful dependence model;
- the target includes population coefficients, group-conditional prediction,
  variance components, or trajectories.

For repeated visits whose residual covariance is the primary object, compare
the [MMRM](mmrm.md) path.

## Declare a model

```python
import pymixef

model = pymixef.Model.from_formula(
    "response ~ treatment * time + (1 + time | subject) + (1 | center)",
    family=pymixef.families.Gaussian(),
)

plan = model.compile(data, engine="lmm", method="reml")
print(plan.explain())
result = plan.fit()
```

Use `||` for independent random coefficients:

```text
response ~ treatment * time + (1 + time || subject)
```

Multiple grouping blocks are supported. Their covariance parameters, levels,
and design columns are compiled explicitly.

The declaration enters through
{py:meth}`pymixef.model.Model.from_formula`, compiles to an
{py:class}`pymixef.model.ExecutionPlan`, and returns a
{py:class}`pymixef.results.FitResult`.

## ML and REML

Under ML, the normalized log likelihood is

$$
\ell(\beta,\theta)=-\frac12
\left[n\log(2\pi)+\log|V_\theta|
+(y-X\beta)^\mathsf{T}V_\theta^{-1}(y-X\beta)\right].
$$

REML additionally accounts for estimating fixed effects by integrating their
linear space under the package’s archived constant convention:

$$
\ell_\mathrm{REML}(\theta)
=-\frac12\left[
(n-p)\log(2\pi)+\log|V_\theta|
+\log|X^\mathsf{T}V_\theta^{-1}X|
+y^\mathsf{T}P_\theta y
\right],
\qquad
P_\theta=V_\theta^{-1}
-V_\theta^{-1}X(X^\mathsf{T}V_\theta^{-1}X)^{-1}X^\mathsf{T}V_\theta^{-1}.
$$

- ML is normally used when comparing nested fixed-effect mean structures.
- REML is commonly used for covariance estimation when the fixed-effect design
  is held constant.
- Do not compare REML objectives from different fixed-effect spaces as if they
  shared the same likelihood.

## What the fit returns

{py:attr}`FitResult.parameters <pymixef.results.FitResult.parameters>` contains
fixed coefficients and natural-scale covariance
summaries such as random-effect SD/correlation and residual SD.
{py:attr}`FitResult.extra <pymixef.results.FitResult.extra>` can include:

- `fixed_effect_names` and `fixed_effect_covariance`;
- objective-convention and likelihood-constant metadata;
- compiled design/audit summaries;
- covariance matrices and optimizer provenance.

{py:attr}`FitResult.random_effects <pymixef.results.FitResult.random_effects>`
contains conditional modes where supported.
{py:attr}`FitResult.fitted_values <pymixef.results.FitResult.fitted_values>` and
{py:attr}`FitResult.residuals <pymixef.results.FitResult.residuals>` align to
analysis rows.

## Prediction modes

```python
conditional = result.prediction(mode="conditional")
population = result.prediction(mode="population")
```

- Conditional prediction includes estimated random effects and generally fits
  observed groups more closely.
- Population prediction uses fixed effects at the random-effect reference value.
- Prediction for new groups requires an explicit new-random-effect simulation or
  integration path; it is not the same as reusing observed group modes.

Both modes are selected explicitly through
{py:meth}`FitResult.prediction <pymixef.results.FitResult.prediction>`.

## Variance and random-effect interpretation

Random-effect SDs describe modeled between-group heterogeneity. A conditional
mode is pulled toward zero according to group information and covariance; it is
not a directly observed group effect. Near-zero variance or near-perfect
correlation can indicate a meaningful boundary, weak information, or an
overly rich random structure.

Inspect
{py:attr}`ConvergenceReport.boundaries <pymixef.convergence.ConvergenceReport.boundaries>`,
the {py:class}`pymixef.convergence.HessianDiagnostics`, scaled gradient, and the
covariance {py:func}`pymixef.covariance.singularity_report` before reporting.

## Diagnostics

At minimum:

1. confirm
   {py:attr}`ConvergenceReport.trustworthy <pymixef.convergence.ConvergenceReport.trustworthy>`;
2. reconcile source and analysis rows;
3. inspect conditional and population fits;
4. plot residuals against fitted values, time, and important predictors;
5. inspect group sizes and random-mode shrinkage;
6. review covariance boundaries/conditioning;
7. run seeded simulation or bootstrap when it answers a prespecified question;
8. test scientifically plausible mean/covariance/missingness sensitivities.

## Reference-engine scope

The initial backend is a dense experimental reference engine intended for
small/moderate correctness experiments. It is not the planned compiled sparse
million-row backend. Passing unit tests or obtaining a trustworthy fit does not
constitute external-software parity for every design.

## Worked examples

- [Catalyst screening](../tutorials/01-catalyst-screening-lmm.md): one batch
  random intercept, adjusted candidate ranking, three diagnostics.
- [Multicenter biomarker](../tutorials/04-multicenter-biomarker-lmm.md): center
  and patient intercepts, treatment-by-week trajectory.
- [Diagnostics/evidence lifecycle](../tutorials/10-diagnostics-simulation-validation-interop-archives.md):
  residuals, VPC, simulation, archives, bundles, and interchange.

## API

- {py:mod}`pymixef.model`
- {py:mod}`pymixef.backends.lmm`
- {py:mod}`pymixef.results`
