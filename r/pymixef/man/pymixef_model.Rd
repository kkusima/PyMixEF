\name{pymixef_model}
\alias{pymixef_model}
\title{Construct a PyMixEF Model from an R Formula}
\usage{
pymixef_model(formula, family = NULL)
}
\arguments{
  \item{formula}{A two-sided R formula in the documented safe,
  lme4-compatible subset.}
  \item{family}{A Python PyMixEF family object, or \code{NULL} for Gaussian.}
}
\value{
A \pkg{reticulate} Python proxy for a \code{pymixef.Model}.
}
\description{
Translates supported R formula text through PyMixEF's compatibility layer and
constructs a Python model without fitting it.
}
\details{
Unsupported R evaluation constructs are rejected before model construction.
The formula's R environment is never executed by PyMixEF. Create transformed
columns explicitly when a transformation is outside the safe formula subset.
}
\examples{
\dontrun{
model <- pymixef_model(y ~ time + (1 | subject))
poisson <- reticulate::import("pymixef")$families$Poisson()
count_model <- pymixef_model(count ~ treatment + (1 | site), poisson)
}
}
\seealso{
\code{\link{pymixef_fit}}
}
\keyword{models}
