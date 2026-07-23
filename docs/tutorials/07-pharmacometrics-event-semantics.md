# Canonical pharmacometrics event semantics

**Field:** pharmacometrics and clinical pharmacology  
**Analysis:** deterministic event canonicalization, ordering, expansion, and audit  
**Records:** dose, observation, covariate, infusion start, and infusion stop

{download}`Download the complete, pre-executed notebook <../../examples/notebooks/07_pharmacometrics_event_semantics.ipynb>`

## Domain and problem

Pharmacometric data are event streams, not merely rectangular numeric tables.
A dose changes a model state, an observation reads a model state, a covariate
record updates context, a reset clears state, and an infusion stop changes the
input rate. These distinctions become especially important when multiple
records share a time or when a compact source row implies additional events.

This tutorial converts ordinary Python mappings into PyMixEF's immutable
{py:class}`pymixef.pharmacometrics.events.EventTable`. It then makes ordering,
`ADDL`/`II` dose expansion, finite-infusion expansion, source-row provenance,
and amount-status semantics inspectable.

For a finite infusion with amount $D$ and duration $\tau$, canonicalization
derives the constant rate and terminal event from

$$
R_\mathrm{in}=\frac{D}{\tau},
\qquad
t_\mathrm{stop}=t_\mathrm{start}+\tau.
$$

For additional doses, the generated time of dose $k$ is
$t_k=t_0+k\,\mathrm{II}$ for $k=1,\ldots,\mathrm{ADDL}$. These equations are
simple; preserving their source-row provenance and same-time priority is the
important part of the event contract.

## What you will learn

By the end of the tutorial, you will be able to:

1. canonicalize dose, observation, covariate, and infusion records;
2. inspect deterministic subject/time/event ordering;
3. understand which state a same-time observation sees;
4. materialize additional doses and generated infusion-stop events;
5. trace every generated row to its source row through the audit log;
6. distinguish a recorded zero dose, an explicitly unknown dose, and an amount
   that is structurally not applicable; and
7. use validation errors as an explicit part of an event-data workflow.

## Dataset and event snapshot

| Item | Value |
| --- | --- |
| Subjects | `P01`, `P02` |
| Source rows | 7 |
| Expanded rows | 10 |
| Generated rows | 3 |
| P01 regimen | 100-unit bolus at 0 h plus 2 additional doses every 12 h |
| P02 regimen | 60-unit infusion over 2 h |
| Derived P02 rate | 30 units/hour |
| P01 observation times | 0 h and 24 h |
| P02 observation times | 0 h and 3 h |
| P01 covariate update | Weight 72 at 12 h |

The example deliberately places observations at the same time as a dose or
infusion start so that ordering is visible rather than implicit.

## Runnable core analysis

The following excerpt reproduces the central event workflow.

```python
from pymixef.pharmacometrics import canonicalize_events

source_records = [
    {
        "ID": "P01",
        "TIME": 0,
        "EVID": "dose",
        "AMT": 100,
        "CMT": "central",
        "ADDL": 2,
        "II": 12,
        "ROW_ID": "dose-p01",
    },
    {
        "ID": "P01",
        "TIME": 0,
        "EVID": 0,
        "DV": 4.8,
        "CMT": "central",
        "ROW_ID": "obs-p01-0",
    },
    {
        "ID": "P01",
        "TIME": 12,
        "EVID": "covariate",
        "COVARIATES": {"weight": 72},
        "ROW_ID": "wt-p01",
    },
    {
        "ID": "P01",
        "TIME": 24,
        "EVID": 0,
        "DV": 1.1,
        "CMT": "central",
        "ROW_ID": "obs-p01-24",
    },
    {
        "ID": "P02",
        "TIME": 0,
        "EVID": "dose",
        "AMT": 60,
        "DUR": 2,
        "CMT": "central",
        "ROW_ID": "inf-p02",
    },
    {
        "ID": "P02",
        "TIME": 0,
        "EVID": 0,
        "DV": 0,
        "CMT": "central",
        "ROW_ID": "obs-p02-0",
    },
    {
        "ID": "P02",
        "TIME": 3,
        "EVID": 0,
        "DV": 2.2,
        "CMT": "central",
        "ROW_ID": "obs-p02-3",
    },
]

events = canonicalize_events(source_records)
same_time_p01 = [
    event.kind
    for event in events
    if event.subject_id == "P01" and event.time == 0
]
assert same_time_p01 == ["dose", "observation"]

expanded = events.expand_additional().expand_infusions()
generated = [event for event in expanded if event.generated]
assert len(expanded) == 10
assert len(generated) == 3
```

## Step-by-step analysis

### 1. Canonicalize heterogeneous source rows

`canonicalize_events(...)` accepts the source mappings, validates them, assigns
event kinds, derives the finite-infusion rate, retains source identifiers, and
returns a deterministically sorted `EventTable`.

The canonical rows are:

| Subject | Time (h) | Kind | Amount | Rate | Row ID |
| --- | ---: | --- | ---: | ---: | --- |
| P01 | 0 | dose | 100 | — | `dose-p01` |
| P01 | 0 | observation | — | — | `obs-p01-0` |
| P01 | 12 | covariate | — | — | `wt-p01` |
| P01 | 24 | observation | — | — | `obs-p01-24` |
| P02 | 0 | infusion start | 60 | 30 | `inf-p02` |
| P02 | 0 | observation | — | — | `obs-p02-0` |
| P02 | 3 | observation | — | — | `obs-p02-3` |

The source row for P02 supplies amount and duration, so PyMixEF derives a
constant rate of `60 / 2 = 30`.

### 2. Inspect the canonical timeline

```{figure} ../_static/tutorials/07_pharmacometrics_event_semantics-figure-1.png
:alt: Two subject timelines showing dose, infusion-start, observation, and covariate events, including offset markers for same-time events at zero hours.
:width: 100%
:name: tutorial-07-canonical-timeline

**Canonical event timeline by subject.** Event type is encoded by color and
marker while time remains on the horizontal axis.
```

**Interpretation.** The timeline exposes both schedules and the same-time
dose/observation pairs at zero. Small vertical offsets separate markers only
for readability; they neither modify the time nor override canonical priority.

### 3. Apply the same-time rule

The complete PyMixEF event priority is:

1. reset;
2. covariate update;
3. infusion stop;
4. bolus dose or infusion start;
5. observation; and
6. other.

Source position is the final stable tie-breaker. Therefore, a same-time
observation in this convention sees the post-dose state. This is a scientific
modeling choice that should be reconciled with the protocol, data derivation,
and intended software comparison.

### 4. Expand compact dosing instructions

Expansion is functional: it returns a new event table and does not mutate the
source table.

```python
expanded = events.expand_additional().expand_infusions()

assert [
    event.kind
    for event in expanded
    if event.subject_id == "P01" and event.time == 24
] == ["dose", "observation"]
```

The expanded schedule adds:

- `dose-p01:addl:1` at 12 h, sourced from `dose-p01`;
- `dose-p01:addl:2` at 24 h, sourced from `dose-p01`; and
- `inf-p02:infusion-stop` at 2 h, sourced from `inf-p02`.

```{figure} ../_static/tutorials/07_pharmacometrics_event_semantics-figure-2.png
:alt: Faceted subject timelines separating original source records from generated additional-dose and infusion-stop records.
:width: 100%
:name: tutorial-07-source-generated

**Canonical source rows and deterministic expansions.** Circles indicate
source rows and stars indicate generated rows.
```

**Interpretation.** P01 gains scheduled doses at 12 and 24 hours, while P02
gains the infusion stop at 2 hours. Every generated event has a deterministic
row identifier and retains the originating `source_row_id`.

### 5. Read provenance from the audit log

The audit describes transformations directly, so a reviewer does not need to
infer them from row-count differences.

```python
from collections import Counter

audit_counts = Counter(entry.code for entry in expanded.audit)
assert audit_counts["EVENT-ADDL-EXPANDED-001"] == 2
assert audit_counts["EVENT-INFUSION-EXPANDED-001"] == 1

generated_audit = [
    entry.to_dict()
    for entry in expanded.audit
    if entry.code
    in {
        "EVENT-ADDL-EXPANDED-001",
        "EVENT-INFUSION-EXPANDED-001",
    }
]
```

The saved audit counts are:

| Audit code | Count | Meaning in this example |
| --- | ---: | --- |
| `EVENT-CANONICALIZED-001` | 7 | One entry for every source row |
| `EVENT-ADDL-EXPANDED-001` | 2 | Two generated P01 doses |
| `EVENT-RATE-DERIVED-001` | 1 | P02 rate derived from amount/duration |
| `EVENT-INFUSION-EXPANDED-001` | 1 | One generated P02 infusion stop |

```{figure} ../_static/tutorials/07_pharmacometrics_event_semantics-figure-3.png
:alt: Horizontal bars showing seven canonicalization, two additional-dose expansion, one rate-derivation, and one infusion-expansion audit records.
:width: 100%
:name: tutorial-07-audit-actions

**Canonicalization and expansion audit actions.** Machine-readable action
counts summarize how the event table was constructed.
```

**Interpretation.** The seven canonicalization records match the seven source
rows. The two additional-dose and one infusion-stop entries match the three
generated rows, while the rate-derived entry records how the infusion input was
resolved.

### 6. Preserve missingness semantics

A missing dose amount is not automatically zero. PyMixEF distinguishes three
states:

```python
from pymixef.pharmacometrics import EventValidationError

unknown = canonicalize_events(
    [{"ID": "P03", "TIME": 0, "EVID": 1, "AMT_STATUS": "unknown"}]
)
zero = canonicalize_events(
    [{"ID": "P03", "TIME": 0, "EVID": 1, "AMT": 0}]
)
observation = canonicalize_events(
    [{"ID": "P03", "TIME": 1, "EVID": 0, "DV": 0.5}]
)

assert (unknown[0].amount, unknown[0].amount_status) == (None, "unknown")
assert (zero[0].amount, zero[0].amount_status) == (0.0, "recorded")
assert (observation[0].amount, observation[0].amount_status) == (
    None,
    "not-applicable",
)

try:
    canonicalize_events([{"ID": "P03", "TIME": 0, "EVID": 1}])
except EventValidationError as error:
    print(error)
```

An ambiguous dose without `AMT` is refused unless
`AMT_STATUS="unknown"` is explicit. The saved message is:

```text
dose records without AMT require AMT_STATUS='unknown'; use AMT=0 for a
recorded zero dose [row=0]
```

## Key saved results

- Canonical subjects: `("P01", "P02")`.
- Original event-table length: `7`.
- Expanded event-table length: `10`.
- Generated-event count: `3`.
- P02 derived infusion rate: `30.0`.
- P01 additional-dose times: `12.0` and `24.0`.
- P02 generated infusion-stop time: `2.0`.
- P01 ordering at 24 h: `["dose", "observation"]`.
- All expansion and same-time assertions passed.
- Unknown dose: amount `None`, status `unknown`.
- Recorded zero dose: amount `0.0`, status `recorded`.
- Observation: amount `None`, status `not-applicable`.

## API map

| Task | Public API | Result used here |
| --- | --- | --- |
| Canonicalize records | {py:func}`pymixef.pharmacometrics.events.canonicalize_events` | Immutable {py:class}`pymixef.pharmacometrics.events.EventTable` |
| Catch invalid event semantics | {py:class}`pymixef.pharmacometrics.events.EventValidationError` | Explicit refusal |
| List subjects | {py:attr}`pymixef.pharmacometrics.events.EventTable.subjects` | Stable subject tuple |
| Iterate/index events | {py:class}`pymixef.pharmacometrics.events.EventTable` sequence interface | {py:class}`pymixef.pharmacometrics.events.CanonicalEvent` objects |
| Expand `ADDL`/`II` | {py:meth}`pymixef.pharmacometrics.events.EventTable.expand_additional` | New event table |
| Expand finite infusions | {py:meth}`pymixef.pharmacometrics.events.EventTable.expand_infusions` | Generated stop events |
| Select one subject | {py:meth}`pymixef.pharmacometrics.events.EventTable.for_subject` | Subject-level event table |
| Inspect transformations | {py:attr}`pymixef.pharmacometrics.events.EventTable.audit` | {py:class}`pymixef.pharmacometrics.events.AuditEntry` sequence |
| Serialize an audit row | {py:meth}`pymixef.pharmacometrics.events.AuditEntry.to_dict` | Plain Python mapping |
| Return ordinary records | {py:meth}`pymixef.pharmacometrics.events.EventTable.to_records` | List of mappings |

Important {py:class}`~pymixef.pharmacometrics.events.CanonicalEvent` fields
used here are `subject_id`, `time`, `kind`,
`amount`, `rate`, `row_id`, `source_row_id`, `generated`, and `amount_status`.

## Exercises

1. Add a reset-and-dose record and inspect its position relative to a same-time
   covariate update.
2. Supply inconsistent `AMT`, `RATE`, and `DUR` values and study the refusal
   message.
3. Use `expanded.for_subject("P01")` to create a subject-only table.
4. Round-trip `expanded.to_records()` through `canonicalize_events(...)` and
   compare subject/time/event ordering.
5. Add two observations at the same subject and time, then verify that source
   position remains the stable tie-breaker.

```{admonition} Interpretation boundaries
:class: important

Use canonicalization as an auditable contract between source data and numerical
modeling. Before simulation, confirm that the documented same-time convention,
units, compartment mapping, and protocol derivations match the intended
analysis. An explicitly unknown amount can be retained for review but must be
resolved before it can drive a numerical state update. The current simulator
also refuses steady-state and modeled or negative infusion-rate semantics
rather than approximating them silently.
```
