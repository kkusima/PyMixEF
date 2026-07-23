\name{pymixef_load}
\alias{pymixef_load}
\title{Load an Archived PyMixEF Result}
\usage{
pymixef_load(path)
}
\arguments{
  \item{path}{A single, non-empty path to a versioned PyMixEF JSON result.}
}
\value{
A \pkg{reticulate} Python proxy for a \code{pymixef.FitResult}.
}
\description{
Loads a versioned JSON result archive using the Python
\code{pymixef.FitResult.load()} method.
}
\details{
This function reads PyMixEF's portable JSON archive format. It does not load
Python pickle files.
}
\examples{
\dontrun{
fit <- pymixef_load("fit-result.json")
fit$convergence$to_dict()
}
}
\seealso{
\code{\link{pymixef_fit}}
}
\keyword{file}
