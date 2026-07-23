skip_if_no_pymixef_python <- function() {
  skip_if_not_installed("reticulate")

  requested <- Sys.getenv("PYMIXEF_PYTHON", unset = "")
  if (nzchar(requested)) {
    reticulate::use_python(requested, required = FALSE)
  }

  if (!reticulate::py_available(initialize = TRUE) ||
      !reticulate::py_module_available("pymixef")) {
    skip("The PyMixEF Python package is unavailable to reticulate.")
  }
}
