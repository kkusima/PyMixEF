# Interoperability and refusal policy

Every import or export returns both a translated value and a
`CompatibilityReport`. Each construct is classified as:

- `exact`;
- `transformed`;
- `approximated`;
- `unsupported`.

Call `result.require_supported()` when a workflow must refuse any unsupported
construct.

The report is part of the result, not console-only commentary. Each
`CompatibilityIssue` names the construct, status, explanation, and source
location when the format exposes one. `CompatibilityReport.by_status(...)`
supports programmatic review.

```python
from pymixef.interoperability import parse_control_stream

translation = parse_control_stream(control_stream_text)
unsupported = translation.report.by_status("unsupported")
for issue in unsupported:
    print(issue.to_dict())

translation.require_supported()  # raises if unsupported issues remain
```

The R-formula translator accepts the safe shared operator subset. NM-TRAN records
are parsed and preserved; arbitrary `$PK`, `$DES`, and `$ERROR` code is not
executed. PharmML and SBML initial support focuses on declarations and emits
explicit limitations for equation constructs outside the subset.

Cross-platform result comparison requires parameter mappings and objective
conventions. It does not infer whether two reported objectives include the same
constants.

## Design principles

1. **Preserve before transforming.** Source text/path and construct locations
   remain available for review where the importer supports them.
2. **Never execute foreign code implicitly.** NM-TRAN structural blocks and R
   environment/function evaluation are not run.
3. **Report each semantic gap.** Unsupported equations, attributes, tasks, or
   duplicate mappings appear in the compatibility report.
4. **Make strictness caller-controlled.** Exploratory inspection can read the
   partial value; production pipelines can call `require_supported()`.
5. **Separate translation from numerical parity.** A compatible model still
   requires explicit result comparison and validation for its intended context.

See the [format-by-format guide](../interoperability/index.md) for the exact
NONMEM, R, PharmML, SBML, and SED-ML subsets.

The `r/pymixef` reticulate interface is alpha software. It routes model
construction and fitting through the same formula translator and has Rd help,
mocked routing tests, and live Python parity tests. `R CMD build` succeeds in the
development environment, and a dependency-complete local
`R CMD check --no-manual` passes. Capability `INT-002` remains gated until a
cross-platform R continuous-integration job produces release evidence and the
deliberately non-routable placeholder maintainer address is replaced for
distribution.
