(api-aliases)=
# API alias reference

PyMixEF keeps several concise, domain-familiar, and compatibility-oriented
names. This page gives every deliberate public alias a searchable anchor and
maps it to the canonical object whose generated API entry contains the full
signature and behavior.

Aliases are not marked as deprecated. The canonical name is used here only to
avoid duplicating the same implementation documentation.

## Root-package conveniences

(alias-pymixef-cov)=
### `pymixef.cov`

**Canonical object:** `pymixef.covariance`  
**Kind:** module alias

`pymixef.cov` and `pymixef.covariance` refer to the same covariance module. See
{doc}`pymixef.covariance <generated/pymixef.covariance>`.

(alias-pymixef-load)=
### `pymixef.load`

**Canonical object:** `pymixef.results.FitResult.load`  
**Kind:** callable bound-classmethod alias

`pymixef.load(path, *, verify_integrity=True, require_sidecar=False)` is bound
directly to `FitResult.load`. It is not an independent module-level function;
loading through either spelling performs the same result deserialization and
integrity checks. See {doc}`pymixef.results <generated/pymixef.results>`.

```python
import pymixef

result = pymixef.load("fit.json")
# Equivalent public spelling:
same_result = pymixef.FitResult.load("fit.json")
```

(alias-pymixef-pattern-mixture-adjust)=
### `pymixef.pattern_mixture_adjust`

**Canonical object:** {py:func}`pymixef.data.pattern_mixture_adjust`  
**Kind:** root-package re-export

Applies an audited response-scale delta only to cells explicitly marked as
imputed and returns `PatternMixtureResult`.

(alias-pymixef-approximation-sensitivity)=
### `pymixef.approximation_sensitivity`

**Canonical object:** {py:func}`pymixef.compare.approximation_sensitivity`  
**Kind:** root-package re-export

Runs named fit-setting scenarios and constructs an aligned parameter/objective
sensitivity table with explicit failure accounting.

(alias-pymixef-group-influence)=
### `pymixef.group_influence`

**Canonical object:** {py:func}`pymixef.diagnostics.group_influence`  
**Kind:** root-package re-export

Deletes complete grouping levels, refits, and optionally compares a
caller-supplied approximation against each full refit.

## Data and formula aliases

(alias-inputadapter)=
### `pymixef.data.InputAdapter`

**Canonical object:** `pymixef.data.adapt_data`  
**Kind:** function alias

Despite its capitalized spelling, `InputAdapter` is the same function object as
`adapt_data`; the object-oriented namespace is the separate `DataAdapter`
class. See {doc}`pymixef.data <generated/pymixef.data>`.

(alias-data-prepare-data)=
### `pymixef.data.prepare_data`

**Canonical object:** `pymixef.data.audit_data`  
**Kind:** function alias

This data-layer name produces `AuditedData`. Do not confuse it with the separate
low-level `pymixef.backends.base.prepare_data` function. See
{doc}`pymixef.data <generated/pymixef.data>`.

(alias-formula-dry-run)=
### `pymixef.formula.dry_run`

**Canonical object:** `pymixef.formula.compile_formula`  
**Kind:** function alias

`dry_run` compiles and audits formula design matrices without fitting. See
{doc}`pymixef.formula <generated/pymixef.formula>`.

## Covariance aliases

(alias-get-covariance)=
### `pymixef.covariance.get_covariance`

**Canonical object:** `pymixef.covariance.covariance_structure`  
**Kind:** function alias

Both names construct a built-in or registered covariance structure by name.
See {doc}`pymixef.covariance <generated/pymixef.covariance>`.

(alias-diagonal-covariance)=
### `pymixef.covariance.DiagonalCovariance`

**Canonical object:** `pymixef.covariance.Diagonal`  
**Kind:** class alias

Both names construct the same diagonal covariance class.

(alias-unstructured-covariance)=
### `pymixef.covariance.UnstructuredCovariance`

**Canonical object:** `pymixef.covariance.Unstructured`  
**Kind:** class alias

Both names construct the same unstructured covariance class.

(alias-compound-symmetry-covariance)=
### `pymixef.covariance.CompoundSymmetryCovariance`

**Canonical object:** `pymixef.covariance.CompoundSymmetry`  
**Kind:** class alias

Both names construct the same compound-symmetry covariance class.

(alias-ar1-covariance)=
### `pymixef.covariance.AR1Covariance`

**Canonical object:** `pymixef.covariance.AR1`  
**Kind:** class alias

Both names construct the same AR(1) covariance class.

(alias-heterogeneous-ar1-covariance)=
### `pymixef.covariance.HeterogeneousAR1Covariance`

**Canonical object:** `pymixef.covariance.HeterogeneousAR1`  
**Kind:** class alias

Both names construct the same heterogeneous AR(1) covariance class.

(alias-toeplitz-covariance)=
### `pymixef.covariance.ToeplitzCovariance`

**Canonical object:** `pymixef.covariance.Toeplitz`  
**Kind:** class alias

Both names construct the same Toeplitz covariance class.

(alias-heterogeneous-toeplitz-covariance)=
### `pymixef.covariance.HeterogeneousToeplitzCovariance`

**Canonical object:** `pymixef.covariance.HeterogeneousToeplitz`  
**Kind:** class alias

Both names construct the same heterogeneous Toeplitz covariance class.

(alias-ante-dependence-covariance)=
### `pymixef.covariance.AnteDependenceCovariance`

**Canonical object:** `pymixef.covariance.AnteDependence`  
**Kind:** class alias

Both names construct the same ante-dependence covariance class.

(alias-spatial-power-covariance)=
### `pymixef.covariance.SpatialPowerCovariance`

**Canonical object:** `pymixef.covariance.SpatialPower`  
**Kind:** class alias

Both names construct the same spatial-power covariance class. Full covariance
class documentation is in
{doc}`pymixef.covariance <generated/pymixef.covariance>`.

## Family class aliases

All aliases in this section are defined by
{doc}`pymixef.families <generated/pymixef.families>`.

(alias-family-normal)=
### `pymixef.families.Normal`

**Canonical object:** `pymixef.families.Gaussian`  
**Kind:** class alias

(alias-family-student)=
### `pymixef.families.Student`

**Canonical object:** `pymixef.families.StudentT`  
**Kind:** class alias

(alias-family-lognormal)=
### `pymixef.families.Lognormal`

**Canonical object:** `pymixef.families.LogNormal`  
**Kind:** class alias

(alias-family-inverse-gauss)=
### `pymixef.families.InverseGauss`

**Canonical object:** `pymixef.families.InverseGaussian`  
**Kind:** class alias

(alias-family-negative-binomial)=
### `pymixef.families.NegativeBinomial`

**Canonical object:** `pymixef.families.NegativeBinomial2`  
**Kind:** class alias

(alias-family-nb1)=
### `pymixef.families.NB1`

**Canonical object:** `pymixef.families.NegativeBinomial1`  
**Kind:** class alias

(alias-family-nb2)=
### `pymixef.families.NB2`

**Canonical object:** `pymixef.families.NegativeBinomial2`  
**Kind:** class alias

(alias-family-gen-poisson)=
### `pymixef.families.GenPoisson`

**Canonical object:** `pymixef.families.GeneralizedPoisson`  
**Kind:** class alias

(alias-family-conway-maxwell-poisson)=
### `pymixef.families.ConwayMaxwellPoisson`

**Canonical object:** `pymixef.families.COMPoisson`  
**Kind:** class alias

(alias-family-zero-inflated)=
### `pymixef.families.ZeroInflatedFamily`

**Canonical object:** `pymixef.families.ZeroInflated`  
**Kind:** class alias

(alias-family-hurdle)=
### `pymixef.families.HurdleFamily`

**Canonical object:** `pymixef.families.Hurdle`  
**Kind:** class alias

(alias-family-truncated)=
### `pymixef.families.TruncatedFamily`

**Canonical object:** `pymixef.families.Truncated`  
**Kind:** class alias

(alias-family-censored)=
### `pymixef.families.CensoredFamily`

**Canonical object:** `pymixef.families.Censored`  
**Kind:** class alias

(alias-family-exponential-survival)=
### `pymixef.families.ExponentialSurvival`

**Canonical object:** `pymixef.families.Exponential`  
**Kind:** class alias

(alias-family-weibull-survival)=
### `pymixef.families.WeibullSurvival`

**Canonical object:** `pymixef.families.Weibull`  
**Kind:** class alias

(alias-family-gompertz-survival)=
### `pymixef.families.GompertzSurvival`

**Canonical object:** `pymixef.families.Gompertz`  
**Kind:** class alias

(alias-family-log-logistic-survival)=
### `pymixef.families.LogLogisticSurvival`

**Canonical object:** `pymixef.families.LogLogistic`  
**Kind:** class alias

## Link namespace aliases

The lowercase `pymixef.families.links` class is a namespace of the same built-in
`Link` objects exposed as uppercase module constants.

(alias-links-identity)=
### `pymixef.families.links.identity`

**Canonical object:** `pymixef.families.IDENTITY`  
**Kind:** constant-object alias

(alias-links-log)=
### `pymixef.families.links.log`

**Canonical object:** `pymixef.families.LOG`  
**Kind:** constant-object alias

(alias-links-logit)=
### `pymixef.families.links.logit`

**Canonical object:** `pymixef.families.LOGIT`  
**Kind:** constant-object alias

(alias-links-probit)=
### `pymixef.families.links.probit`

**Canonical object:** `pymixef.families.PROBIT`  
**Kind:** constant-object alias

(alias-links-cloglog)=
### `pymixef.families.links.cloglog`

**Canonical object:** `pymixef.families.CLOGLOG`  
**Kind:** constant-object alias

(alias-links-cauchit)=
### `pymixef.families.links.cauchit`

**Canonical object:** `pymixef.families.CAUCHIT`  
**Kind:** constant-object alias

(alias-links-inverse)=
### `pymixef.families.links.inverse`

**Canonical object:** `pymixef.families.INVERSE`  
**Kind:** constant-object alias

(alias-links-inverse-squared)=
### `pymixef.families.links.inverse_squared`

**Canonical object:** `pymixef.families.INVERSE_SQUARED`  
**Kind:** constant-object alias

## Family method compatibility names

These are callable method spellings on `pymixef.families.Family`. The forwarding
methods retain the same family-specific probability or random-generation
semantics as their canonical counterparts.

(alias-family-log-probability)=
### `pymixef.families.Family.log_probability`

**Canonical operation:** `pymixef.families.Family.log_prob`  
**Kind:** callable forwarding-method alias

(alias-family-logpdf)=
### `pymixef.families.Family.logpdf`

**Canonical operation:** `pymixef.families.Family.log_prob`  
**Kind:** callable SciPy-style forwarding-method alias

(alias-family-logpmf)=
### `pymixef.families.Family.logpmf`

**Canonical object:** `pymixef.families.Family.logpdf`  
**Kind:** callable method alias

(alias-family-random)=
### `pymixef.families.Family.random`

**Canonical operation:** `pymixef.families.Family.rvs`  
**Kind:** callable forwarding-method alias

## Backend aliases

(alias-backend-lmm)=
### `pymixef.backends.LMMBackend`

**Canonical object:** `pymixef.backends.lmm.GaussianLMMBackend`  
**Kind:** class alias

The same alias is also available as `pymixef.backends.lmm.LMMBackend`. See
{doc}`pymixef.backends.lmm <generated/pymixef.backends.lmm>`.

(alias-backend-lmm-owning-module)=
### `pymixef.backends.lmm.LMMBackend`

**Canonical object:** `pymixef.backends.lmm.GaussianLMMBackend`  
**Kind:** class alias

This is the owning-module spelling of the `pymixef.backends.LMMBackend`
re-export.

(alias-backend-dense-lmm)=
### `pymixef.backends.DenseLMMBackend`

**Canonical object:** `pymixef.backends.lmm.GaussianLMMBackend`  
**Kind:** class alias

The same alias is also available as `pymixef.backends.lmm.DenseLMMBackend`.

(alias-backend-dense-lmm-owning-module)=
### `pymixef.backends.lmm.DenseLMMBackend`

**Canonical object:** `pymixef.backends.lmm.GaussianLMMBackend`  
**Kind:** class alias

This is the owning-module spelling of the
`pymixef.backends.DenseLMMBackend` re-export.

(alias-backend-glmm)=
### `pymixef.backends.GLMMBackend`

**Canonical object:** `pymixef.backends.glmm.LaplaceGLMMBackend`  
**Kind:** class alias

The same alias is also available as `pymixef.backends.glmm.GLMMBackend`. See
{doc}`pymixef.backends.glmm <generated/pymixef.backends.glmm>`.

(alias-backend-glmm-owning-module)=
### `pymixef.backends.glmm.GLMMBackend`

**Canonical object:** `pymixef.backends.glmm.LaplaceGLMMBackend`  
**Kind:** class alias

This is the owning-module spelling of the `pymixef.backends.GLMMBackend`
re-export.

(alias-estimated-marginal-means)=
### `pymixef.backends.mmrm.estimated_marginal_means`

**Canonical object:** `pymixef.backends.mmrm.linear_inference`  
**Kind:** function alias

(alias-mmrm-contrasts)=
### `pymixef.backends.mmrm.contrasts`

**Canonical object:** `pymixef.backends.mmrm.linear_inference`  
**Kind:** function alias

Both MMRM names expose the same labeled linear-inference calculation. See
{doc}`pymixef.backends.mmrm <generated/pymixef.backends.mmrm>`.

## IR and error aliases

(alias-ir-schema-version)=
### `pymixef.ir.IR_SCHEMA_VERSION`

**Canonical constant:** `pymixef.ir.MODEL_IR_SCHEMA_VERSION`  
**Kind:** constant alias

Both names contain the current model-IR schema version. See
{doc}`pymixef.ir <generated/pymixef.ir>`.

(alias-model-ir-hash)=
### `pymixef.ir.ModelIR.hash`

**Canonical property:** `pymixef.ir.ModelIR.semantic_hash`  
**Kind:** property alias

Both properties return the SHA-256 digest of the canonical mathematical
representation. Neither spelling is called with parentheses.

(alias-model-diff)=
### `pymixef.ir.model_diff`

**Canonical object:** `pymixef.ir.diff_models`  
**Kind:** function alias

Both names compare two `ModelIR` objects and return a `ModelDiff`.

(alias-unsupported-engine-error)=
### `pymixef.errors.UnsupportedEngineError`

**Canonical object:** `pymixef.errors.EngineCompatibilityError`  
**Kind:** exception-class alias

Both names identify the same exception class. See
{doc}`pymixef.errors <generated/pymixef.errors>`.

## Pharmacometric declaration aliases

(alias-pharmacometrics-d)=
### `pymixef.pharmacometrics.d`

**Canonical operation:** `pymixef.pharmacometrics.derivative`  
**Kind:** callable short-form wrapper

`d(state, expression)` delegates to `derivative(state, expression)`. Both are
defined by {doc}`pymixef.pharmacometrics.dsl
<generated/pymixef.pharmacometrics.dsl>`.

(alias-pharmacometrics-d-owning-module)=
### `pymixef.pharmacometrics.dsl.d`

**Canonical operation:** `pymixef.pharmacometrics.dsl.derivative`  
**Kind:** callable short-form wrapper

This is the owning-module spelling of the `pymixef.pharmacometrics.d`
re-export.

(alias-pharmacometrics-saem)=
### `pymixef.pharmacometrics.saem`

**Canonical object:** `pymixef.pharmacometrics.experimental_saem`  
**Kind:** function alias

The alias is also available as `pymixef.pharmacometrics.estimation.saem`. It is
the same callback-driven experimental kernel and does not imply an integrated
population estimator. See {doc}`pymixef.pharmacometrics.estimation
<generated/pymixef.pharmacometrics.estimation>`.

(alias-pharmacometrics-saem-owning-module)=
### `pymixef.pharmacometrics.estimation.saem`

**Canonical object:** `pymixef.pharmacometrics.estimation.experimental_saem`  
**Kind:** function alias

This is the owning-module spelling of the `pymixef.pharmacometrics.saem`
re-export.

## Closed-form PK aliases

The following aliases are defined by
{doc}`pymixef.pharmacometrics.pk <generated/pymixef.pharmacometrics.pk>` and are
also re-exported from `pymixef.pharmacometrics`.

(alias-one-compartment-bolus)=
### `pymixef.pharmacometrics.one_compartment_bolus`

**Canonical object:** `pymixef.pharmacometrics.one_compartment_iv_bolus`  
**Kind:** function alias

The owning-module spelling is
`pymixef.pharmacometrics.pk.one_compartment_bolus`.

(alias-one-compartment-bolus-owning-module)=
### `pymixef.pharmacometrics.pk.one_compartment_bolus`

**Canonical object:** `pymixef.pharmacometrics.pk.one_compartment_iv_bolus`  
**Kind:** function alias

(alias-one-compartment-iv-infusion)=
### `pymixef.pharmacometrics.one_compartment_iv_infusion`

**Canonical object:** `pymixef.pharmacometrics.one_compartment_infusion`  
**Kind:** function alias

The owning-module spelling is
`pymixef.pharmacometrics.pk.one_compartment_iv_infusion`.

(alias-one-compartment-iv-infusion-owning-module)=
### `pymixef.pharmacometrics.pk.one_compartment_iv_infusion`

**Canonical object:** `pymixef.pharmacometrics.pk.one_compartment_infusion`  
**Kind:** function alias

(alias-two-compartment-bolus)=
### `pymixef.pharmacometrics.two_compartment_bolus`

**Canonical object:** `pymixef.pharmacometrics.two_compartment_iv_bolus`  
**Kind:** function alias

The owning-module spelling is
`pymixef.pharmacometrics.pk.two_compartment_bolus`.

(alias-two-compartment-bolus-owning-module)=
### `pymixef.pharmacometrics.pk.two_compartment_bolus`

**Canonical object:** `pymixef.pharmacometrics.pk.two_compartment_iv_bolus`  
**Kind:** function alias

(alias-two-compartment-iv-infusion)=
### `pymixef.pharmacometrics.two_compartment_iv_infusion`

**Canonical object:** `pymixef.pharmacometrics.two_compartment_infusion`  
**Kind:** function alias

The owning-module spelling is
`pymixef.pharmacometrics.pk.two_compartment_iv_infusion`.

(alias-two-compartment-iv-infusion-owning-module)=
### `pymixef.pharmacometrics.pk.two_compartment_iv_infusion`

**Canonical object:** `pymixef.pharmacometrics.pk.two_compartment_infusion`  
**Kind:** function alias
