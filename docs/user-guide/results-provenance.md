# Results, reporting, and provenance

`FitResult` is the durable boundary between estimation and downstream analysis.
It stores numerical output together with the information needed to interpret
and reproduce it.

## Result contents

Common fields include:

- `engine`, `method`, objective, and log likelihood;
- immutable parameter mapping;
- `ConvergenceReport`;
- fitted values, residuals, and conditional random effects where supported;
- backend-specific `extra` values such as coefficient covariance and visit
  covariance;
- structured diagnostics;
- the compiled `ModelIR`;
- `RunManifest` with data/model fingerprints, environment, options, warnings,
  timing, and reproducibility class.

Convenience properties include `success` and `n_observations`. Use
`summary()` for human review and `to_dict()` for structured processing.

## Portable result archives

```python
path = result.save("result.json")
loaded = pymixef.FitResult.load(path)

assert loaded.parameters == result.parameters
assert loaded.manifest.model_ir_hash == result.manifest.model_ir_hash
```

Archives are JSON rather than pickle, so the payload is inspectable and does not
execute arbitrary Python during load. A `.sha256` sidecar detects content drift.
Hashes provide integrity evidence, not identity, approval, confidentiality, or a
digital signature.

`pymixef.load` is a convenience alias for `FitResult.load`.

## Model and data fingerprints

`fingerprint_model_ir` hashes canonical semantic model content.
`fingerprint_data` hashes the normalized analysis data contract.
`environment_snapshot` records runtime context. Equivalent semantic IR should
have the same semantic hash even when nonsemantic formatting differs.

Model hashes help answer “did the model change?” They do not establish that a
model is identifiable or appropriate.

## Reports

```python
from pymixef import render_report

render_report(result, "report.md")
render_report(result, "report.html")
render_report(result, "report.pdf")
render_report(result, "report.docx")
```

The file suffix chooses Markdown, HTML, PDF, or Word. Install the `report` extra
for optional formats. A generated report presents retained evidence; it does not
replace scientific review or an organization’s controlled approval process.

## Validation bundles

```python
bundle = pymixef.create_validation_bundle(
    result,
    "analysis-evidence",
    include_data=False,
)
verification = pymixef.verify_validation_bundle(bundle)
assert verification["valid"]
```

A bundle contains a manifest, result, traceability material, and a human-readable
README. Raw data are excluded by default; include data only with authorization
and an appropriate privacy/security assessment.

`verify_validation_bundle` checks file integrity and internal consistency.
`traceability_matrix`, `classify_change`, and `change_impact` connect
requirements, implementation areas, tests, and change review.

## Comparison evidence

Use `compare` to preserve mapping and conventions when checking PyMixEF output
against another implementation. A close numerical comparison supports a
particular model/data/software configuration; it should not be generalized
beyond the tested space without evidence.

## Reproducibility classes

The manifest’s `ReproducibilityClass` distinguishes deterministic behavior from
calculations that require tolerances or recorded random seeds. Reproducibility
means rerunning the declared computation under controlled conditions; it is not
the same as replicating a scientific result on independent data.

## API map

- [`pymixef.results`](../api/generated/pymixef.results.rst)
- [`pymixef.provenance`](../api/generated/pymixef.provenance.rst)
- [`pymixef.reporting`](../api/generated/pymixef.reporting.rst)
- [`pymixef.validation`](../api/generated/pymixef.validation.rst)
- [Validation and traceability policy](../validation.md)
