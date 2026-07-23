# Internal import boundary. Keeping this in one binding makes the wrapper
# independently testable without importing Python during API-routing tests.
.pymixef_import <- function() {
  reticulate::import("pymixef", delay_load = FALSE)
}

.pymixef_formula_text <- function(formula) {
  if (!inherits(formula, "formula") || length(formula) != 3L) {
    stop("`formula` must be a two-sided R formula.", call. = FALSE)
  }
  paste(deparse(formula, width.cutoff = 500L), collapse = " ")
}

.pymixef_translate_formula <- function(py, formula_text) {
  translation <- py$interoperability$translate_r_formula(
    formula_text
  )
  translation$report$require_supported()
  translation$value
}

#' Construct a PyMixEF model from an R formula
#'
#' @param formula A two-sided R formula in the documented safe,
#'   lme4-compatible subset.
#' @param family A Python PyMixEF family object, or `NULL` for Gaussian.
#'
#' @return A Python `pymixef.Model` proxy.
#'
#' @details
#' Formula text is translated by PyMixEF's compatibility layer. Unsupported R
#' evaluation constructs are rejected before model construction. The formula's
#' R environment is never executed by PyMixEF.
#' @export
pymixef_model <- function(formula, family = NULL) {
  formula_text <- .pymixef_formula_text(formula)
  py <- .pymixef_import()
  formula_text <- .pymixef_translate_formula(py, formula_text)
  if (is.null(family)) {
    family <- py$families$Gaussian()
  }
  py$Model$from_formula(formula_text, family = family)
}

#' Fit a PyMixEF formula model
#'
#' @param formula A two-sided R formula in the supported subset.
#' @param data An R `data.frame`. Column names referenced by `formula` must be
#'   present.
#' @param ... Named arguments passed to Python's `pymixef.fit()`, such as
#'   `method`, `engine`, or engine-specific settings.
#'
#' @return A Python `pymixef.FitResult` proxy.
#'
#' @details
#' R data frames are converted by `reticulate`. Estimation, convergence
#' assessment, and archival behavior are implemented by the Python package.
#' @export
pymixef_fit <- function(formula, data, ...) {
  if (!is.data.frame(data)) {
    stop("`data` must be an R data.frame.", call. = FALSE)
  }
  formula_text <- .pymixef_formula_text(formula)
  py <- .pymixef_import()
  formula_text <- .pymixef_translate_formula(py, formula_text)
  py$fit(formula_text, data = data, ...)
}

#' Load an archived PyMixEF result
#'
#' @param path A single, non-empty path to a versioned PyMixEF JSON result.
#'
#' @return A Python `pymixef.FitResult` proxy.
#'
#' @details
#' This function loads PyMixEF's versioned JSON archive. It does not read Python
#' pickle files.
#' @export
pymixef_load <- function(path) {
  if (!is.character(path) || length(path) != 1L || is.na(path) ||
      !nzchar(path)) {
    stop("`path` must be one non-empty character string.", call. = FALSE)
  }
  py <- .pymixef_import()
  py$FitResult$load(path)
}
