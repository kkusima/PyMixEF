# Public validation artifacts

PyMixEF keeps the version-controlled source of truth for requirement coverage in
`src/pymixef/capabilities.py`. Generate the complete, machine-readable
requirement-to-source-to-test matrix from any installed release with:

```bash
pymixef traceability > traceability.json
```

Each row states its implementation status, maturity, reproducibility class,
source module, specification document, evidence, and known limitations. A true
implementation status applies only to the narrowly named 0.1 capability; it is
not a blueprint stage-gate result. The generated matrix is also embedded in
every validation bundle created by `pymixef bundle`.

These artifacts support context-of-use validation. They are evidence records,
not independent verification reports, a universal regulatory approval, or a
compliance certificate.
