# Validation, traceability, and evidence bundles

PyMixEF provides machine-readable building blocks for a context-specific
validation process:

- a capability-to-source-to-specification traceability matrix;
- deterministic change classification and recommended test reruns;
- portable validation evidence bundles; and
- bundle member and hash verification.

These tools preserve evidence and make boundaries visible. They do not declare
a model scientifically appropriate, approve an analysis, qualify an
organization's computerized system, or turn an experimental package into a
universally validated product.

```{admonition} Regulatory and scientific scope
:class: important

PyMixEF 0.1 is an **experimental reference implementation**. An
`implemented=true` capability means that the deliberately narrow behavior named
by that capability exists and has the recorded evidence. It does not mean that
external-software parity has been established or that the package is suitable
for every clinical, regulated, or decision-bearing use.
```

## Public API at a glance

The complete module-level validation surface covered on this page is:

```python
from pymixef.validation import (
    RequirementLinks,
    TRACEABILITY_LINKS,
    TraceabilityRecord,
    change_impact,
    classify_change,
    create_validation_bundle,
    traceability_matrix,
    verify_validation_bundle,
)
```

The exact callable signatures are:

```text
RequirementLinks(
    source_files: tuple[str, ...],
    specification_files: tuple[str, ...],
)

TraceabilityRecord(
    requirement: str,
    capability: str,
    stage: str,
    maturity: str,
    implemented: bool,
    reproducibility: str | None,
    source_files: tuple[str, ...],
    specification_files: tuple[str, ...],
    tests: tuple[str, ...],
    evidence: tuple[str, ...],
    limitations: tuple[str, ...] = (),
)

traceability_matrix() -> tuple[TraceabilityRecord, ...]

classify_change(
    path: str | os.PathLike[str],
) -> str

change_impact(
    paths: Iterable[str | os.PathLike[str]],
) -> dict[str, Any]

create_validation_bundle(
    result: FitResult,
    path: str | Path,
    *,
    analysis_data: Any | None = None,
    include_data: bool = False,
    additional_files: Sequence[str | Path] = (),
) -> Path

verify_validation_bundle(
    path: str | Path,
) -> dict[str, Any]
```

The four workflow functions `traceability_matrix`, `change_impact`,
`create_validation_bundle`, and `verify_validation_bundle` are also exported
from the top-level `pymixef` package. The link and record types, mapping
constant, and single-path classifier are accessed from
`pymixef.validation`.

## A practical evidence model

It helps to keep four questions separate:

1. **What was intended?** A requirement identifier and its specification files.
2. **Where is it implemented?** Source files linked to that requirement.
3. **What evidence exists?** Tests, reference calculations, reports, or other
   version-controlled artifacts.
4. **What claim is justified?** The capability's stage, maturity,
   implementation flag, reproducibility class, and interpretation boundaries.

Traceability makes those links inspectable. A bundle preserves a snapshot of
the links alongside a particular result. Neither mechanism decides whether the
requirement or analysis is adequate for a proposed use.

## Requirement links

### `RequirementLinks`

`RequirementLinks` is an immutable, slotted dataclass with two tuple fields:

| Field | Meaning |
| --- | --- |
| `source_files` | Repository-relative implementation paths |
| `specification_files` | Repository-relative documents specifying the supported scope |

For example:

```python
from pymixef.validation import RequirementLinks

links = RequirementLinks(
    source_files=("src/pymixef/backends/lmm.py",),
    specification_files=("docs/methods/lmm.md",),
)
```

The type records links only. It does not contain maturity, evidence, or an
implementation decision.

### `TRACEABILITY_LINKS`

`TRACEABILITY_LINKS` is a
`dict[str, RequirementLinks]` keyed by stable capability/requirement
identifiers such as `ARCH-001`, `LMM-001`, `REG-001`, and `VAL-001`.

The current repository enforces these invariants in
`tests/test_validation.py`:

- the mapping keys exactly equal the capability identifiers in
  `pymixef.capabilities.CAPABILITIES`;
- every traceability record has at least one specification file;
- every implemented record has at least one Python source file; and
- linked source and specification paths are concrete, repository-relative
  files without parent-directory traversal.

An entry for a gated requirement may point to partial primitives or may have an
empty source tuple. The capability's `implemented` flag and recorded
interpretation boundaries remain authoritative; the existence of a link is not
an implementation claim.

Inspect an individual mapping directly:

```python
from pymixef.validation import TRACEABILITY_LINKS

lmm_links = TRACEABILITY_LINKS["LMM-001"]
print(lmm_links.source_files)
print(lmm_links.specification_files)
```

## Traceability records and matrix

### `TraceabilityRecord`

`TraceabilityRecord` is an immutable, slotted dataclass representing one
complete requirement row.

| Field | Meaning |
| --- | --- |
| `requirement` | Stable capability identifier |
| `capability` | Human-readable capability name |
| `stage` | Capability or workflow stage |
| `maturity` | Current maturity label, such as `experimental` |
| `implemented` | Whether the narrowly defined behavior is implemented |
| `reproducibility` | Reproducibility class string, or `None` for a gated row |
| `source_files` | Linked implementation modules |
| `specification_files` | Linked scope/specification documents |
| `tests` | Evidence entries beginning with `tests/` |
| `evidence` | Remaining evidence not classified as a test, Python source, or specification |
| `limitations` | Recorded interpretation boundaries or open work |

`TraceabilityRecord.to_dict()` returns all eleven fields as a JSON-compatible
mapping. Tuple-valued fields are converted to lists.

### `traceability_matrix()`

`traceability_matrix()` walks the capability registry in registry order and
returns a tuple containing one `TraceabilityRecord` per capability.

Its classification rules are deterministic:

- capability evidence beginning with `tests/` becomes `tests`;
- evidence beginning with `src/` or `benchmarks/` and ending in `.py` is merged
  into `source_files`;
- source paths from `TRACEABILITY_LINKS` come first and duplicates are removed;
- specification paths come from `TRACEABILITY_LINKS`; and
- all remaining capability evidence stays in `evidence`.

The current registry contains 56 rows: 42 narrowly implemented capabilities
and 14 gated capabilities.

```python
from pymixef.validation import traceability_matrix

records = traceability_matrix()
implemented = [record for record in records if record.implemented]
gated = [record for record in records if not record.implemented]

assert len(records) == 56
assert len(implemented) == 42
assert len(gated) == 14

lmm = next(
    record
    for record in records
    if record.requirement == "LMM-001"
)
print(lmm.to_dict())
```

For installed-version automation, generate the same rows as JSON:

```bash
pymixef traceability > traceability.json
```

The command has no `--json` switch because JSON is already its only output
format.

```{admonition} A matrix is a map, not a verdict
:class: note

The matrix answers where requirements, implementation, tests, and stated open
work are recorded. Reviewers still need to assess test independence, coverage
of the intended model/data space, acceptance criteria, deviations, and whether
the evidence is sufficient for the declared context of use.
```

## Change-impact classification

### `classify_change()`

`classify_change(path)` converts the supplied path to POSIX form and applies the
following rules in order. Supply repository-relative paths so the prefix rules
match.

| First matching path prefix | Returned category |
| --- | --- |
| `docs/`, `README`, `CONTRIBUTING`, or `GOVERNANCE` | `documentation-only` |
| `src/pymixef/backends/` or `src/pymixef/families.py` | `statistical-method` |
| `src/pymixef/pharmacometrics/` | `numerical` |
| `pyproject.toml`, `requirements`, or `uv.lock` | `dependency` |
| `.github/` or `SECURITY` | `security-or-build` |
| any other `src/` path | `api` |
| anything else | `other` |

Examples:

```python
from pymixef.validation import classify_change

assert (
    classify_change("docs/validation.md")
    == "documentation-only"
)
assert (
    classify_change("src/pymixef/backends/lmm.py")
    == "statistical-method"
)
assert (
    classify_change("src/pymixef/pharmacometrics/ode.py")
    == "numerical"
)
assert (
    classify_change("src/pymixef/results.py")
    == "api"
)
```

### `change_impact()`

`change_impact(paths)` classifies every supplied path, groups paths by category,
and returns:

```python
{
    "classifications": {
        "category": ["path", "..."],
    },
    "recommended_reruns": ["test-or-suite", "..."],
}
```

The built-in rerun logic is exact and intentionally compact:

| Trigger | Recommendations added |
| --- | --- |
| Every call, including an empty path list | `tests/test_provenance.py` |
| At least one `statistical-method` or `numerical` path | `tests/test_families.py`, `tests/test_lmm.py`, `tests/test_glmm_mmrm.py`, `tests/test_ode_pk.py` |
| At least one `api` path | `tests/test_ir.py`, `tests/test_model.py`, `tests/test_results.py` |
| At least one `dependency` path | `full cross-platform suite` |

Recommendations are deduplicated and sorted. Multiple categories produce the
union of their recommendations. The current rules add no category-specific
rerun beyond the always-present provenance test for `documentation-only`,
`security-or-build`, or `other`.

```python
from pymixef.validation import change_impact

impact = change_impact(
    [
        "docs/validation.md",
        "src/pymixef/pharmacometrics/ode.py",
        "src/pymixef/results.py",
        "pyproject.toml",
    ]
)

assert impact == {
    "classifications": {
        "documentation-only": ["docs/validation.md"],
        "numerical": [
            "src/pymixef/pharmacometrics/ode.py"
        ],
        "api": ["src/pymixef/results.py"],
        "dependency": ["pyproject.toml"],
    },
    "recommended_reruns": [
        "full cross-platform suite",
        "tests/test_families.py",
        "tests/test_glmm_mmrm.py",
        "tests/test_ir.py",
        "tests/test_lmm.py",
        "tests/test_model.py",
        "tests/test_ode_pk.py",
        "tests/test_provenance.py",
        "tests/test_results.py",
    ],
}
```

### Change-impact workflow

A controlled workflow can use these results as follows:

1. Collect repository-relative changed paths from the reviewed change set.
2. Store the output of `change_impact(...)` with the change record.
3. Run every recommended test or suite.
4. Add project-specific tests based on the actual semantic change, risk, and
   context of use.
5. Review whether specifications, capability states, warnings, examples, or
   acceptance criteria also need revision.
6. Preserve commands, environments, outputs, deviations, and review decisions.

`change_impact(...)` does not inspect Git, execute tests, understand semantic
diffs, or assert that its minimal recommendations are sufficient. It is a
deterministic starting point for a risk-based review, not a replacement for
one. There is currently no change-impact CLI command.

## Validation evidence bundles

### What `create_validation_bundle()` writes

`create_validation_bundle(...)` creates a ZIP archive and returns the
destination `Path`. For the same in-memory result and the same supplied data and
attachment bytes, archive construction is deterministic:

- member names are written in sorted order;
- member timestamps are fixed to `2020-01-01 00:00:00`;
- members use DEFLATE compression;
- archived permission bits are fixed to `0644`; and
- canonical JSON is used for result, manifest, traceability, and optional data.

A newly fitted result can still differ across runs because the result manifest
contains run-specific information such as creation time, elapsed time, and
environment. Bundle determinism does not erase those scientifically relevant
differences.

The creator always writes:

| Member | Exact content |
| --- | --- |
| `result.json` | Canonical JSON from `result.to_dict()`, followed by a newline |
| `manifest.json` | Canonical JSON from `result.manifest.to_dict()`, followed by a newline |
| `traceability.json` | Canonical JSON list from the current `traceability_matrix()`, followed by a newline |
| `README.txt` | Human-readable data-inclusion statement and context-specific-validation notice |
| `SHA256SUMS.json` | Sorted mapping of member name to bare hexadecimal SHA-256 digest |

`SHA256SUMS.json` hashes every member assembled before it, but it does not hash
itself.

Conditional members are:

| Condition | Member written |
| --- | --- |
| `analysis_data` supplied and `include_data=True` | `analysis-data.json` containing canonical JSON |
| Each path in `additional_files` | `attachments/<basename>` containing the file's original bytes |

If an attachment path is not a file, `create_validation_bundle(...)` raises
`FileNotFoundError`. Parent directories for the destination are created
automatically.

### Data inclusion and privacy behavior

The data behavior is deliberately explicit:

| Inputs | Bundle behavior | README statement |
| --- | --- | --- |
| `analysis_data=None` | No `analysis-data.json` | No raw analysis data were supplied |
| Data supplied, `include_data=False` | No `analysis-data.json` | Raw data were intentionally excluded; the run manifest input fingerprint is named |
| Data supplied, `include_data=True` | Canonical `analysis-data.json` included | Raw analysis data are included as canonical JSON |
| `include_data=True`, but `analysis_data=None` | No `analysis-data.json` | No raw analysis data were supplied |

`include_data=False` is the default.

```{admonition} Excluding the input table is not de-identification
:class: warning

`result.json` is always included and can contain fitted values, residuals,
random effects, row-aligned diagnostic data, warnings, and metadata.
`additional_files` are copied as raw bytes regardless of `include_data`.
Review the complete archive for identifiers, sensitive derived values,
confidential attachments, and destination controls. Only include raw or
sensitive material after explicit authorization and an appropriate
privacy/security assessment.
```

The bundle is not encrypted. SHA-256 digests provide integrity evidence, not
confidentiality.

### Runnable Python example

This example creates a small fitted result, excludes the supplied raw table, and
verifies the resulting archive:

```python
from pathlib import Path

import numpy as np

import pymixef

rng = np.random.default_rng(20260710)
subject_index = np.repeat(np.arange(10), 4)
time = np.tile(np.arange(4, dtype=float), 10)
random_intercept = rng.normal(0.0, 0.6, 10)
y = (
    2.0
    + 0.5 * time
    + random_intercept[subject_index]
    + rng.normal(0.0, 0.3, len(time))
)
data = {
    "y": y,
    "time": time,
    "subject": np.array(
        [f"S{i + 1:02d}" for i in subject_index]
    ),
}

result = pymixef.fit(
    "y ~ time + (1 | subject)",
    data=data,
    method="reml",
    compute_hessian=False,
)

bundle_path = pymixef.create_validation_bundle(
    result,
    Path("artifacts") / "validation-bundle.zip",
    analysis_data=data,
    include_data=False,
)

verification = pymixef.verify_validation_bundle(
    bundle_path
)
assert verification["valid"] is True
assert "result.json" in verification["files"]
assert "analysis-data.json" not in verification["files"]
```

To include an authorized synthetic dataset and an existing analysis plan:

```python
from pymixef.validation import create_validation_bundle

bundle_path = create_validation_bundle(
    result,
    "artifacts/authorized-evidence.zip",
    analysis_data=data,
    include_data=True,
    additional_files=("analysis-plan.md",),
)
```

The `analysis-plan.md` path must exist. Apply the same authorization review to
attachments that you apply to analysis data.

## Bundle verification

### What `verify_validation_bundle()` checks

`verify_validation_bundle(path)` opens the ZIP and:

1. reads `SHA256SUMS.json`;
2. detects duplicate member names;
3. requires the actual member set to equal the names in the checksum mapping
   plus `SHA256SUMS.json`;
4. rejects unexpected or missing members;
5. recalculates SHA-256 for every member named in the mapping;
6. reads `manifest.json`; and
7. returns the archived manifest and sorted verified file names when all
   declared hashes match.

The successful return contract is:

```python
{
    "valid": True,
    "manifest": {...},
    "files": [
        "README.txt",
        "manifest.json",
        "result.json",
        "traceability.json",
        # optional analysis-data.json and attachments
    ],
}
```

The `files` list contains members covered by `SHA256SUMS.json`; it does not
include `SHA256SUMS.json` itself.

### Structured verification failures

Defined member-set failures raise `pymixef.ValidationError` with:

| Code | Trigger | Structured details |
| --- | --- | --- |
| `VALIDATION-BUNDLE-MEMBERS-001` | Duplicate, unexpected, or missing members relative to the checksum mapping | `duplicates`, `unexpected`, and `missing` lists |
| `VALIDATION-BUNDLE-HASH-001` | One or more member bytes do not match the declared SHA-256 | Per-file expected and observed digests |

```python
from pymixef.errors import ValidationError
from pymixef.validation import verify_validation_bundle

try:
    verification = verify_validation_bundle(
        "validation-bundle.zip"
    )
except ValidationError as error:
    print(error.code)
    print(error.details)
else:
    assert verification["valid"]
```

A malformed ZIP, invalid checksum JSON, or missing checksum/manifest that
cannot be read may instead raise the corresponding Python `zipfile`, key, or
JSON exception. The verifier expects a PyMixEF bundle container.

### What verification does not claim

The verifier does **not**:

- authenticate the archive's author or approver;
- digitally sign or encrypt the archive;
- anchor `SHA256SUMS.json` to an external trusted digest;
- prevent a party from changing both a member and its checksum;
- load `result.json` as a `FitResult` or re-run result-level semantic checks;
- compare the separate manifest with the manifest embedded in `result.json`;
- re-run the analysis or reproduce the recorded environment;
- test whether external traceability paths still exist;
- inspect attachments for safety or sensitive information; or
- establish scientific correctness, model adequacy, regulatory acceptance, or
  fitness for a context of use.

Use an approved external signing, access-control, retention, and review process
when authenticity and governance are required.

## Command-line workflow

### Export capability and traceability records

```bash
pymixef capabilities --json > capabilities.json
pymixef traceability > traceability.json
```

### Fit, bundle, and verify

```bash
pymixef fit \
  "y ~ time + (1 | subject)" \
  --data analysis.csv \
  --family gaussian \
  --engine lmm \
  --method reml \
  --output result.json

pymixef bundle \
  result.json \
  --output validation-bundle.zip

pymixef verify-bundle \
  validation-bundle.zip \
  > bundle-verification.json
```

The `bundle` command loads the result with `FitResult.load(...)`, creates a
default bundle, prints the destination, and does not expose CLI flags for raw
data or attachments. The result integrity sidecar is checked when present but
is not required by this command.

`verify-bundle` prints the successful verification mapping as JSON. CLI
argument/container errors and `ValidationError` failures propagate as nonzero
command failures; there is no separate documented bundle-specific exit-code
table.

For authorized data inclusion or attachments, use the Python API so the choice
is explicit in code review.

## Context-of-use validation workflow

PyMixEF's artifacts can support an organization's controlled validation
process. A practical workflow is:

1. **Define intended use.** State the users, decisions, model classes, data
   characteristics, environments, interfaces, and consequences of error.
2. **Select applicable requirements.** Start from `traceability_matrix()` and
   record which rows apply, which do not, and why.
3. **Specify acceptance criteria.** Define numerical tolerances,
   convergence expectations, failure behavior, reference conventions,
   reproducibility controls, and review thresholds before running comparisons.
4. **Assess evidence independence.** Distinguish implementation-unit tests,
   same-package cross-checks, independently implemented analytic checks,
   external-software comparisons, and user-acceptance evidence.
5. **Execute in the qualified environment.** Preserve package and dependency
   versions, platform, numerical-library/thread settings, seeds, commands,
   inputs or authorized fingerprints, outputs, and deviations.
6. **Review scientific adequacy.** Evaluate design, estimand, likelihood and
   covariance assumptions, missingness, diagnostics, approximation sensitivity,
   and uncertainty independently of software execution success.
7. **Bundle relevant evidence.** Include only authorized data and attachments;
   verify the archive and retain any external signature or approval record.
8. **Control change.** Run `change_impact(...)`, add risk-based tests, and
   reassess the validation state whenever software, dependencies, methods,
   data flow, or intended use changes.

The context-of-use statement determines how much evidence is needed. A teaching
notebook, exploratory laboratory model, clinical-trial analysis, and regulated
production workflow do not have the same risk or acceptance criteria.

## Current validation evidence

The capability registry is the authoritative machine-readable status. The
validation-specific rows currently state:

| ID | Capability | Stage | Implemented | Maturity | Reproducibility | Recorded evidence |
| --- | --- | --- | --- | --- | --- | --- |
| `REG-001` | Validation bundle generator | regulated-workflow-support | `True` | experimental | bitwise | `tests/test_validation.py` |
| `REG-002` | Change-impact classification | regulated-workflow-support | `True` | experimental | bitwise | `tests/test_validation.py` |
| `VAL-001` | Public traceability matrix | foundation | `True` | experimental | bitwise | `tests/test_validation.py` |
| `VAL-002` | Selected independent reference calculations | foundation | `True` | experimental | deterministic-with-tolerance | `tests/test_families.py` |
| `VAL-003` | Initial failure and pathology corpus | foundation | `True` | experimental | deterministic-with-tolerance | `tests/test_formula.py`, `tests/test_data_covariance.py`, `tests/test_glmm_mmrm.py` |

The present repository evidence includes:

- normalized Gaussian, Student-t, Poisson, and binomial likelihood comparisons
  against SciPy for selected parameterizations;
- normalization and moment checks for additional family objects;
- one-compartment IV-bolus ODE agreement with a same-package closed form;
- one analytic exponential-decay sensitivity spot check plus forward/central
  finite-difference diagnostics;
- representative parser, data, covariance, engine, event, convergence,
  serialization, and integrity failure tests;
- result round-trips, ModelIR/output-hash checks, validation-bundle checks, and
  an unlisted-member rejection test;
- explicit comparison-convention tests and report wording;
- ten assertion-backed notebooks with clean-kernel replay validation; and
- a reduced synthetic LMM benchmark harness that records environment and
  result data.

The repository defines CI across Ubuntu, macOS, and Windows for Python 3.11,
3.12, and 3.13, plus separate notebook and documentation jobs. The primary
matrix runs tests with a 70% coverage floor, Ruff, mypy, the reduced benchmark,
and package construction. A workflow definition is an automated gate; archived
successful run records are the evidence that a particular revision passed it.

`VAL-002` is intentionally narrow: current external reference checks cover
selected normalized family likelihoods, not independent full-engine or
cross-software parity reports. `VAL-003` is an initial corpus rather than broad
adversarial and platform-specific testing.

## Current open gates

The 14 currently gated capability rows are:

| ID | Open capability | Current boundary |
| --- | --- | --- |
| `ARCH-003` | Full backend conformance suite | Fit-payload cases cover every built-in backend; separate objective, gradient, optional Hessian-vector product, and simulation contracts remain absent from the Backend Protocol |
| `LMM-002` | Sparse million-row LMM engine | Requires the planned compiled sparse backend |
| `GLMM-002` | Adaptive Gauss-Hermite quadrature | Rejected until an order-sensitivity suite exists |
| `SAEM` | Integrated SAEM population estimator | Research callback kernel is not connected to ModelIR, event/error compilation, population diagnostics, or stable `FitResult` |
| `INT-002` | Release-gated thin R wrapper | Local build/check evidence exists, but no cross-platform R CI gate; maintainer address is a non-routable placeholder |
| `LMM-003` | Profile, bootstrap, and robust LMM inference | Generic restartable bootstrap exists; profile and sandwich paths remain gated |
| `GLMM-004` | glmmTMB Salamanders parity | Zero-inflated NB2 parity remains a later evidence gate |
| `NLME-005` | Finite-mixture population estimation | Weights are representable; label-stable population fitting is unavailable |
| `ODE-003` | Independently validated ODE sensitivities | No analytic/automatic sensitivities, event-discontinuity corpus, or independent multi-model suite |
| `EST-002` | Independent derivative verification suite | No separate derivative implementation and systematic cross-engine verification suite |
| `ADV-001` | Backend-neutral priors exported to two samplers | Priors are represented; two-backend equivalence is not validated |
| `ADV-002` | Robust sensitivity comparison | No robust-likelihood fit path or automated cross-model report |
| `ADV-003` | Joint longitudinal-event simulation | No joint model, shared random-effect simulator, or validated event-time likelihood |
| `PERF-003` | Explicit numerical thread controls | Thread environment is recorded, but numerical-library thread counts are not configured or enforced |

Additional evidence priorities include:

- broader analytic reference calculations;
- derivative verification implemented independently of estimator code;
- matched-convention cross-software reports for supported engines;
- simulation-recovery studies with Monte Carlo uncertainty;
- wider adversarial/pathology corpora;
- cross-platform numerical and performance regression reports; and
- documented review and approval evidence for each intended context of use.

Other unavailable methods are rejected rather than silently replaced by a
different estimator.

## Evidence-package checklist

For a decision-bearing use, consider retaining:

- intended use, risk assessment, and applicable requirements;
- package source/version and immutable distribution hashes;
- dependency lock or environment inventory;
- model specification, ModelIR, formula/declaration, and data contract;
- authorized inputs or controlled input fingerprints;
- commands, numerical settings, seeds, and thread configuration;
- convergence, warnings, diagnostics, estimates, uncertainty, and simulation
  results;
- independent references with explicit parameter/objective conventions;
- acceptance criteria and signed review outcomes;
- traceability snapshot and change-impact assessment;
- verified evidence bundle plus any approved external signature;
- deviations, investigations, and residual-risk decisions; and
- retention, access-control, and privacy records.

This checklist is a starting point. Applicable organizational procedures,
quality systems, laws, regulations, and scientific standards determine the
actual record.

## Related documentation

- [Capability catalog](reference/capability-catalog.md)
- [Results, reports, and provenance](user-guide/results-provenance.md)
- [Command-line reference](reference/cli.md)
- [Warning catalog](warnings.md)
- [Generated `pymixef.validation` API](api/generated/pymixef.validation.rst)
- [Executed diagnostics and evidence tutorial](tutorials/10-diagnostics-simulation-validation-interop-archives.md)
