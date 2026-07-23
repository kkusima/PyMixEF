\name{pymixef-package}
\alias{pymixef}
\alias{pymixef-package}
\docType{package}
\title{R Interface to PyMixEF}
\description{
A thin \pkg{reticulate} interface to the PyMixEF Python package. The wrapper
uses the Python package's versioned model representation, compatibility checks,
fit engines, and result archives.
}
\details{
The wrapper is an alpha interface. It accepts a restricted lme4-style formula
subset and does not evaluate arbitrary R formula environments. Unsupported
constructs are refused by the PyMixEF compatibility layer.

The Python package named \code{pymixef} must be installed in the environment
selected by \pkg{reticulate}. The package metadata intentionally uses the
non-routable placeholder \email{maintainers@example.invalid} until project
governance assigns an accountable public contact.
}
\seealso{
\code{\link{pymixef_model}}, \code{\link{pymixef_fit}},
\code{\link{pymixef_load}}
}
\keyword{package}
