\name{pymixef_fit}
\alias{pymixef_fit}
\title{Fit a PyMixEF Formula Model}
\usage{
pymixef_fit(formula, data, ...)
}
\arguments{
  \item{formula}{A two-sided R formula in the supported subset.}
  \item{data}{An R \code{data.frame}. Columns referenced by \code{formula}
  must be present.}
  \item{...}{Named arguments passed to Python's \code{pymixef.fit()}, such as
  \code{method}, \code{engine}, or engine-specific settings.}
}
\value{
A \pkg{reticulate} Python proxy for a \code{pymixef.FitResult}.
}
\description{
Translates a supported formula and fits it through the PyMixEF Python engine.
}
\details{
R data frames are converted by \pkg{reticulate}. Estimation, convergence
assessment, diagnostics, and archival behavior are implemented by the Python
package. Unsupported formula constructs are refused before fitting.
}
\examples{
\dontrun{
fit <- pymixef_fit(
  y ~ time + (1 | subject),
  data = trial_data,
  method = "reml"
)
fit$convergence$to_dict()
}
}
\seealso{
\code{\link{pymixef_model}}, \code{\link{pymixef_load}}
}
\keyword{models}
