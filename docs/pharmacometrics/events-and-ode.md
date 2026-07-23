# Event records and ODE simulation

Pharmacometric calculation depends on exact event semantics. PyMixEF first
canonicalizes records, then expands deterministic schedule constructs, and only
then applies events to an ODE state.

## Canonicalize records

```python
from pymixef.pharmacometrics import canonicalize_events

events = canonicalize_events(
    [
        {"ID": "P01", "TIME": 0.0, "EVID": 1, "AMT": 100.0, "CMT": 1},
        {"ID": "P01", "TIME": 0.0, "EVID": 0, "DV": 5.0, "MDV": 0},
        {"ID": "P01", "TIME": 12.0, "EVID": 0, "DV": 1.2, "MDV": 0},
    ]
)

print(events.subjects)
print(events.to_records())
```

Accepted fields cover subject, time, event type, dose amount/rate/duration,
compartment, additional doses/interval, steady-state indicator, observation and
missing-dependent-variable fields, quantification limit, occasion,
bioavailability, lag, covariates, and extra source fields.

## Stable same-time priority

Events at the same subject/time are ordered:

1. reset;
2. covariate update;
3. infusion stop;
4. bolus dose or infusion start;
5. observation;
6. other event.

Source position is a stable tie-breaker. Therefore a same-time observation sees
the post-dose state. Confirm that this matches the data convention and protocol;
the software does not infer a study’s intended predose/postdose meaning.

`EVENT_PRIORITY` in {py:mod}`pymixef.pharmacometrics.events` exposes the
ordering contract. {py:class}`pymixef.pharmacometrics.events.CanonicalEvent` fields
`row_id`, `source_row_id`, `source_position`, `generated`, and `generation` preserve
lineage.

## Amount semantics

These states are different:

- **recorded zero**: amount is `0.0`, status `recorded`;
- **explicit unknown**: amount is `None`, status `unknown`;
- **structurally not applicable**: observations/covariates have no dose amount.

An ambiguous missing dose amount raises
{py:class}`pymixef.pharmacometrics.events.EventValidationError`. This prevents a
missing value from becoming a silent zero.

## Additional doses and infusions

Canonical tables are immutable. Expansion returns a new table:

```python
expanded = events.expand_additional().expand_infusions()

for entry in expanded.audit:
    print(entry.to_dict())
```

- `ADDL` plus `II` creates individually identifiable generated dose rows.
- A finite infusion creates a generated stop event.
- If amount and duration are supplied, rate may be derived and audited.
- Generated rows retain `source_row_id`.

For a constant finite infusion, the derived rate and stop time are

$$
R_\mathrm{in}=\frac{D}{\tau},
\qquad
t_\mathrm{stop}=t_\mathrm{start}+\tau.
$$

You can request expansion during canonicalization:

```python
events = canonicalize_events(
    records,
    expand_additional=True,
    expand_infusions=True,
)
```

Steady-state records and modeled or negative infusion rates are represented
conservatively and unsupported simulation semantics are refused explicitly.

## ODE right-hand side

The ODE callback receives time, state, and
{py:class}`pymixef.pharmacometrics.ode.ODEContext`. The context contains
parameters, time-varying covariates, active infusion rates, and subject ID.

Between events, {py:func}`pymixef.pharmacometrics.ode.simulate_ode` solves the
general initial-value problem

$$
\frac{dx(t)}{dt}=f\{t,x(t),\theta,c(t),u(t)\},
\qquad x(t_0)=x_0,
$$

where $c(t)$ is covariate context and $u(t)$ is the active infusion input.
At event time $t_e$, an exact action map $x(t_e^+)=\mathcal E_e\{x(t_e^-)\}$
is applied before the next continuous segment.

```python
import numpy as np
from pymixef.pharmacometrics import simulate_ode

def rhs(time, state, context):
    amount = state[0]
    clearance = context.parameters["CL"]
    volume = context.parameters["V"]
    infusion = context.infusion_rates[0]
    return np.array([infusion - clearance * amount / volume])

simulation = simulate_ode(
    rhs,
    initial_state=[0.0],
    events=events,
    t_eval=np.linspace(0.0, 24.0, 97),
    parameters={"CL": 5.0, "V": 20.0},
    state_names=("central",),
    compartment_map={1: "central"},
    sensitivity_parameters=("CL",),
)

amount = simulation.state("central")
d_amount_d_cl = simulation.sensitivity("CL", state="central")
print(simulation.metadata)
```

The state is whatever the callback defines. A common central state is amount,
not concentration; divide by volume when that is the observation mapping.
PyMixEF does not automatically convert units.

## Discontinuities and segments

The integrator advances continuously between event times, applies discontinuous
state actions exactly at their event boundary, and resumes integration. Solver
metadata records method, SciPy version, tolerances, maximum step, success,
function/Jacobian/LU counts, segments, event actions, source/generated event
counts, and same-time order.

## Sensitivities

{py:func}`pymixef.pharmacometrics.ode.simulate_ode` calculates sensitivities using
the supported numerical path and records method/step in metadata.
{py:func}`pymixef.pharmacometrics.ode.finite_difference_sensitivities` can
compare forward and central finite differences and returns a
{py:class}`pymixef.pharmacometrics.ode.SensitivityCheck`.

For parameter $\theta_k$, the central approximation is

$$
\frac{\partial x(t)}{\partial\theta_k}
\approx
\frac{x(t;\theta_k+h)-x(t;\theta_k-h)}{2h}.
$$

Sensitivities are local derivatives around the supplied parameter values. Their
scale and numerical step matter; they do not establish parameter
identifiability.

## Multiple subjects

{py:func}`pymixef.pharmacometrics.ode.simulate_subjects` splits a canonical
table by subject and returns a mapping from subject ID to
{py:class}`pymixef.pharmacometrics.ode.ODESimulationResult`. Each result retains event snapshots
and subject-specific metadata.

## Failures are structured

- {py:class}`pymixef.pharmacometrics.events.EventValidationError`: malformed or ambiguous canonical data.
- {py:class}`pymixef.pharmacometrics.ode.UnsupportedEventSemantics`: a well-formed event requests behavior the
  simulator does not implement.
- {py:class}`pymixef.pharmacometrics.ode.ODESimulationError`: integration or numerical event processing fails, with
  time/subject/details when available.

## API and tutorial

- {py:mod}`pymixef.pharmacometrics.events`
- {py:mod}`pymixef.pharmacometrics.ode`
- [Executed event-semantics tutorial](../tutorials/07-pharmacometrics-event-semantics.md)
- [Executed PK/ODE cross-check](../tutorials/08-closed-form-pk-and-ode.md)
