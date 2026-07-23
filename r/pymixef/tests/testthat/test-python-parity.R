test_that("safe R formulas preserve Python model semantics", {
  skip_if_no_pymixef_python()

  py <- reticulate::import("pymixef", delay_load = FALSE)
  formula <- y ~ time + treatment + (1 | subject)
  formula_text <- pymixef:::.pymixef_formula_text(formula)
  translated <- py$interoperability$translate_r_formula(formula_text)

  expect_true(isTRUE(translated$report$supported))
  model <- pymixef_model(formula)
  expect_identical(model$formula_text(), translated$value)

  ir <- model$to_ir()$to_dict()
  expect_identical(ir$metadata$authoring_surface, "formula")
  expect_identical(ir$response, "y")
})

test_that("unsupported R evaluation is refused consistently", {
  skip_if_no_pymixef_python()

  data <- data.frame(
    y = c(1, 2, 3),
    x = c(0, 1, 2),
    subject = c("a", "b", "c")
  )
  unsafe <- y ~ poly(x, 2) + (1 | subject)

  expect_error(pymixef_model(unsafe), "unsupported")
  expect_error(pymixef_fit(unsafe, data), "unsupported")
})

test_that("the R fit surface returns the shared result contract", {
  skip_if_no_pymixef_python()

  data <- data.frame(
    subject = rep(sprintf("s%02d", 1:8), each = 3),
    time = rep(0:2, times = 8)
  )
  subject_effect <- rep(seq(-0.7, 0.7, length.out = 8), each = 3)
  data$y <- 2 + 0.4 * data$time + subject_effect +
    rep(c(-0.08, 0.03, 0.05), times = 8)

  result <- pymixef_fit(
    y ~ time + (1 | subject),
    data,
    method = "ml",
    maxiter = 200L
  )

  expect_identical(result$engine, "lmm")
  expect_identical(result$method, "ml")
  expect_equal(result$n_observations, nrow(data))
  expect_true(is.logical(result$success))
  expect_true(is.list(result$convergence$to_dict()))
  expect_true(is.list(result$manifest$to_dict()))
})
