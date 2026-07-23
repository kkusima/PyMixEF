# Closed-form PK and observation error

Closed-form helpers calculate concentrations for linear one- and
two-compartment models from supplied parameters. They are useful for simulation,
unit tests, and cross-checking event-aware ODE implementations.

## One-compartment routes

```python
import numpy as np
from pymixef.pharmacometrics import (
    one_compartment_infusion,
    one_compartment_iv_bolus,
    one_compartment_oral,
)

time = np.linspace(0.0, 24.0, 97)

bolus = one_compartment_iv_bolus(
    time, dose=100.0, clearance=5.0, volume=20.0
)
infusion = one_compartment_infusion(
    time,
    dose=100.0,
    duration=2.0,
    clearance=5.0,
    volume=20.0,
)
oral = one_compartment_oral(
    time,
    dose=100.0,
    clearance=5.0,
    volume=20.0,
    absorption_rate=1.2,
)
```

Bolus accepts bioavailability and lag. Infusion accepts a consistent
dose/rate/duration declaration plus start time. Oral accepts first-order
absorption, bioavailability, and lag.

With $k=CL/V$, the one-compartment bolus and first-order oral solutions are

$$
C_\mathrm{bolus}(t)=\frac{FD}{V}e^{-k(t-t_\mathrm{lag})},
$$

$$
C_\mathrm{oral}(t)
=\frac{FDk_a}{V(k_a-k)}
\left[e^{-k(t-t_\mathrm{lag})}-e^{-k_a(t-t_\mathrm{lag})}\right],
\qquad t\ge t_\mathrm{lag}.
$$

{py:class}`pymixef.pharmacometrics.pk.OneCompartmentPK` stores the structural
parameters for an object-oriented form.
{py:func}`pymixef.pharmacometrics.pk.one_compartment_bolus` and
{py:func}`pymixef.pharmacometrics.pk.one_compartment_iv_infusion` are aliases.

## Two-compartment routes

{py:func}`pymixef.pharmacometrics.pk.two_compartment_iv_bolus`,
{py:func}`pymixef.pharmacometrics.pk.two_compartment_infusion`, and
{py:func}`pymixef.pharmacometrics.pk.two_compartment_oral` accept clearance,
central volume, intercompartmental
clearance, and peripheral volume. `two_compartment_rates` returns the derived
micro/rate constants `k10`, `k12`, `k21`, `alpha`, and `beta`.

The microconstants and hybrid exponents satisfy

$$
k_{10}=\frac{CL}{V_c},\qquad
k_{12}=\frac{Q}{V_c},\qquad
k_{21}=\frac{Q}{V_p},
$$

$$
\alpha,\beta
=\frac12\left[
k_{10}+k_{12}+k_{21}
\pm\sqrt{(k_{10}+k_{12}+k_{21})^2-4k_{10}k_{21}}
\right].
$$

```python
from pymixef.pharmacometrics import two_compartment_rates

rates = two_compartment_rates(
    clearance=5.0,
    central_volume=20.0,
    intercompartmental_clearance=3.0,
    peripheral_volume=35.0,
)
print(rates)
```

{py:class}`pymixef.pharmacometrics.pk.TwoCompartmentPK` bundles the four
structural parameters. Bolus/infusion
aliases are indexed in the API alias page.

## Parameter and unit contracts

- clearances, volumes, doses, absorption rates, and required durations/rates
  are validated for their mathematical domains;
- scalar or array-like time is accepted;
- units are not converted—time, rate, dose, volume, and clearance must be
  mutually consistent;
- functions return structural concentration, not noisy observations;
- a same-package closed-form/ODE agreement test is regression evidence, not an
  independent implementation comparison.

Invalid parameters raise
{py:class}`pymixef.pharmacometrics.pk.PKValidationError`.

## Observation-error models

Let $f$ be structural prediction. Error objects define conditional variance
and likelihood behavior:

| Factory/class | Variance form |
|---|---|
| {py:func}`~pymixef.pharmacometrics.pk.additive` / {py:class}`~pymixef.pharmacometrics.pk.AdditiveError` | $\sigma_a^2$ |
| {py:func}`~pymixef.pharmacometrics.pk.proportional` / {py:class}`~pymixef.pharmacometrics.pk.ProportionalError` | $(\sigma_p f)^2$ |
| {py:func}`~pymixef.pharmacometrics.pk.power` / {py:class}`~pymixef.pharmacometrics.pk.PowerError` | $\sigma_p^2|f|^{2p}$ |
| {py:func}`~pymixef.pharmacometrics.pk.combined` / {py:class}`~pymixef.pharmacometrics.pk.CombinedError` | $\sigma_a^2+\sigma_p^2|f|^{2p}$ |
| {py:func}`~pymixef.pharmacometrics.pk.lognormal` / {py:class}`~pymixef.pharmacometrics.pk.LogNormalError` | error on log concentration scale |

```python
from pymixef.pharmacometrics import combined

error = combined(0.10, 0.15)
variance = error.variance(bolus)
```

The parameters are supplied values; calling `variance` does not estimate them.
Error objects can also compute log likelihood and generate random observations
through their shared protocol.

## Censoring helpers

{py:func}`pymixef.pharmacometrics.pk.left_censored_loglikelihood`,
{py:func}`pymixef.pharmacometrics.pk.right_censored_loglikelihood`, and
{py:func}`pymixef.pharmacometrics.pk.interval_censored_loglikelihood` combine a
prediction with an {py:class}`pymixef.pharmacometrics.pk.ObservationError` and
censoring limits. They use CDF/survival/interval
probability rather than replacing censored observations with a fixed fraction
of the limit.

```python
from pymixef.pharmacometrics import left_censored_loglikelihood

log_likelihood = left_censored_loglikelihood(
    limit=[0.1, 0.1],
    prediction=[0.08, 0.15],
    error=error,
)
```

## API and tutorial

- {py:mod}`pymixef.pharmacometrics.pk`
- [Executed closed-form and ODE tutorial](../tutorials/08-closed-form-pk-and-ode.md)
