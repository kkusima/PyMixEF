# pymixef

`pymixef` is the experimental R interface to the PyMixEF Python package. It
uses `reticulate` and sends supported mixed-effects formula text to the same
versioned model representation and engines used by Python.

```r
library(pymixef)

model <- pymixef_model(y ~ time + (1 | subject))
fit <- pymixef_fit(
  y ~ time + (1 | subject),
  data = trial_data,
  method = "reml"
)
```

The Python package must be installed in the Python environment selected by
`reticulate`. For a dedicated environment, select it before loading or calling
the wrapper:

```r
reticulate::use_python("/path/to/python", required = TRUE)
library(pymixef)
```

## Formula boundary

The wrapper accepts a deliberately restricted, safe lme4-style formula subset.
It sends formula text to PyMixEF's compatibility checker; it does not evaluate
the formula environment in R. Constructs that depend on R evaluation, including
namespace access, `$`, `[[`, `I()`, `poly()`, `ns()`, and `bs()`, are refused.
Create transformed columns explicitly before fitting.

## Alpha status and contact limitation

This wrapper is an alpha interface and is not ready for CRAN submission or
regulated production use. Its creator address,
`maintainers@example.invalid`, is deliberately non-routable: project governance
has not yet assigned a public maintainer identity or support channel. This
placeholder is kept explicit rather than attributing the package to a fictional
person. Downstream distributors must replace it with an accountable contact
before publishing the package.

PyMixEF provides no warranty of fitness for clinical, regulatory, or other
high-stakes decisions. Review the Python package's capability and validation
artifacts before relying on a calculation path.
