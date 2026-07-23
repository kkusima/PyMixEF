test_that("the public API is explicit and stable", {
  expect_true(all(
    c("pymixef_fit", "pymixef_load", "pymixef_model") %in%
      getNamespaceExports("pymixef")
  ))
  expect_named(formals(pymixef_model), c("formula", "family"))
  expect_named(formals(pymixef_fit), c("formula", "data", "..."))
  expect_named(formals(pymixef_load), "path")
})

test_that("R-side argument errors happen before Python import", {
  expect_error(
    pymixef_model("y ~ x"),
    "`formula` must be a two-sided R formula.",
    fixed = TRUE
  )
  expect_error(
    pymixef_fit(y ~ x, list(y = 1, x = 2)),
    "`data` must be an R data.frame.",
    fixed = TRUE
  )
  expect_error(
    pymixef_load(character()),
    "`path` must be one non-empty character string.",
    fixed = TRUE
  )
  expect_error(
    pymixef_load(NA_character_),
    "`path` must be one non-empty character string.",
    fixed = TRUE
  )
})

test_that("model construction and fit route through the same translation", {
  translated <- character()
  supported_checks <- 0L
  model_calls <- list()
  fit_calls <- list()

  fake_python <- list(
    interoperability = list(
      translate_r_formula = function(value) {
        translated <<- c(translated, value)
        list(
          value = value,
          report = list(require_supported = function() {
            supported_checks <<- supported_checks + 1L
            invisible(NULL)
          })
        )
      }
    ),
    families = list(Gaussian = function() "gaussian-family"),
    Model = list(from_formula = function(value, family) {
      model_calls <<- list(value = value, family = family)
      "model-result"
    }),
    fit = function(value, data, ...) {
      fit_calls <<- list(value = value, data = data, settings = list(...))
      "fit-result"
    }
  )
  local_mocked_bindings(
    .pymixef_import = function() fake_python,
    .package = "pymixef"
  )

  formula <- y ~ time + treatment + (1 | subject)
  data <- data.frame(
    y = c(1, 2),
    time = c(0, 1),
    treatment = c("A", "B"),
    subject = c("s1", "s2")
  )

  expect_identical(pymixef_model(formula), "model-result")
  expect_identical(
    pymixef_fit(formula, data, method = "ml"),
    "fit-result"
  )

  expect_length(unique(translated), 1L)
  expect_identical(model_calls$value, fit_calls$value)
  expect_identical(model_calls$family, "gaussian-family")
  expect_identical(fit_calls$data, data)
  expect_identical(fit_calls$settings$method, "ml")
  expect_identical(supported_checks, 2L)
})

test_that("load routes through the versioned FitResult loader", {
  loaded_path <- NULL
  fake_python <- list(
    FitResult = list(load = function(path) {
      loaded_path <<- path
      "loaded-result"
    })
  )
  local_mocked_bindings(
    .pymixef_import = function() fake_python,
    .package = "pymixef"
  )

  expect_identical(pymixef_load("analysis.json"), "loaded-result")
  expect_identical(loaded_path, "analysis.json")
})
