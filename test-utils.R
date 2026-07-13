# ---------------------------------------------------------------------------
# Unit tests for the TVF-SurvFlow *utility* functions.
#
# These intentionally cover only the functions that depend on base R + dplyr,
# so they run WITHOUT satpred / keras / tensorflow / pcoxtime etc. The model
# wrappers need the full modelling stack (and a Python backend for the deep
# models), so smoke tests for those are sketched at the bottom and skipped by
# default — enable them once the environment is set up.
#
# HOW TO RUN
#   1. install.packages(c("testthat", "dplyr"))
#   2. source("TVF-SurvFlow_functions.R")   # or the functions below
#   3. testthat::test_dir("tests/testthat")
#
# NOTE: these tests were drafted by reading the code and have not been executed
# in an R session here — expect to fix small details (e.g. exact column names)
# on first run.
# ---------------------------------------------------------------------------

library(testthat)

# If you have refactored the utilities into their own file, source it here.
# Otherwise, sourcing the full functions file will also load the heavy
# libraries at the top — for pure-unit testing you may prefer to copy the four
# utility functions into a small helper file and source that instead.
# source("../../TVF-SurvFlow_functions.R")

# --- a tiny synthetic counting-process data set used across tests -----------
make_toy <- function() {
  data.frame(
    id     = c(1, 1, 2, 2, 3, 3, 4, 5),
    tstart = c(0, 5, 0, 3, 0, 4, 0, 0),
    tstop  = c(5, 9, 3, 8, 4, 7, 6, 5),
    status = c(0, 1, 0, 1, 0, 1, 1, 0),
    age    = c(30, 30, 41, 41, 55, 55, 47, 62),
    treat  = factor(c("A", "A", "B", "B", "A", "A", "B", "A")),
    time   = c(9, 9, 8, 8, 7, 7, 6, 5)
  )
}

# ---------------------------------------------------------------------------
context("create_output_folder")

test_that("create_output_folder creates the three standard subfolders", {
  tmp <- file.path(tempdir(), paste0("tvf_test_", as.integer(runif(1, 1, 1e6))))
  on.exit(unlink(tmp, recursive = TRUE), add = TRUE)

  paths <- create_output_folder(tmp)

  expect_true(dir.exists(file.path(tmp, "models")))
  expect_true(dir.exists(file.path(tmp, "summaries_for_each_model")))
  expect_true(dir.exists(file.path(tmp, "metrics")))
  expect_named(paths, c("models", "summaries_for_each_model", "metrics"))
})

test_that("create_output_folder is idempotent (no error on re-run)", {
  tmp <- file.path(tempdir(), paste0("tvf_test_", as.integer(runif(1, 1, 1e6))))
  on.exit(unlink(tmp, recursive = TRUE), add = TRUE)

  create_output_folder(tmp)
  expect_error(create_output_folder(tmp), NA)  # NA = expect no error
})

# ---------------------------------------------------------------------------
context("split_data")

test_that("split_data keeps all rows of a subject on the same side", {
  df <- make_toy()
  sp <- split_data(df, train_prop = 0.6, seed = 42)

  train_ids <- unique(sp$train$id)
  test_ids  <- unique(sp$test$id)

  # no subject appears in both train and test
  expect_length(intersect(train_ids, test_ids), 0)
  # every original subject ends up somewhere
  expect_setequal(union(train_ids, test_ids), unique(df$id))
})

test_that("split_data is reproducible for a fixed seed", {
  df <- make_toy()
  a <- split_data(df, train_prop = 0.6, seed = 7)
  b <- split_data(df, train_prop = 0.6, seed = 7)
  expect_equal(a$train, b$train)
  expect_equal(a$test,  b$test)
})

test_that("split_data returns a train and test element", {
  df <- make_toy()
  sp <- split_data(df, train_prop = 0.8, seed = 1)
  expect_named(sp, c("train", "test"))
  expect_s3_class(sp$train, "data.frame")
  expect_s3_class(sp$test,  "data.frame")
})

# ---------------------------------------------------------------------------
context("replace_with_na")

test_that("replace_with_na leaves excluded columns untouched", {
  df <- make_toy()
  out <- replace_with_na(df, perc = 50, exclude = c("status", "time"))
  expect_false(any(is.na(out$status)))
  expect_false(any(is.na(out$time)))
})

test_that("replace_with_na introduces missing values in non-excluded columns", {
  df <- make_toy()
  set.seed(123)
  out <- replace_with_na(df, perc = 100, exclude = c("status", "time"))
  # at 100% every eligible column should contain at least one NA
  eligible <- setdiff(names(df), c("status", "time"))
  expect_true(any(vapply(out[eligible], function(col) any(is.na(col)), logical(1))))
})

# ---------------------------------------------------------------------------
context("replace_and_impute_data")

test_that("replace_and_impute_data returns a data frame with no NAs introduced by imputation", {
  df <- make_toy()
  set.seed(99)
  out <- replace_and_impute_data(df, perc = 30)
  expect_s3_class(out, "data.frame")
  # numeric and factor columns should be imputed (no NAs remaining in them)
  expect_false(any(is.na(out$age)))
})

# ---------------------------------------------------------------------------
context("clean_vimp_df")

test_that("clean_vimp_df normalises terms/Overall naming", {
  vi <- data.frame(terms = c("age", "treat"), Overall = c(0.4, 0.1))
  out <- clean_vimp_df(vi)
  expect_true(all(c("Variable", "Importance") %in% names(out)))
  expect_equal(nrow(out), 2)
})

test_that("clean_vimp_df normalises type/importance naming", {
  vi <- data.frame(type = c("age", "treat"), importance = c(0.7, 0.2))
  out <- clean_vimp_df(vi)
  expect_true(all(c("Variable", "Importance") %in% names(out)))
})

test_that("clean_vimp_df returns NULL on NULL input", {
  expect_null(clean_vimp_df(NULL))
})

# ---------------------------------------------------------------------------
# OPTIONAL smoke tests for the modelling layer — require the full stack.
# Remove the skip() lines once satpred / pcoxtime / etc. are installed.
# ---------------------------------------------------------------------------
context("model wrappers (integration, skipped by default)")

test_that("coxph_func returns a fit and a concordance", {
  skip("Enable once the full modelling stack + global formulas are set up.")

  df <- make_toy()
  # the model functions read these from the global env:
  var <- setdiff(colnames(df), c("id", "tstart", "tstop", "status", "time"))
  formula_with_tvf_without_cluster    <<- as.formula(paste("Surv(tstart, tstop, status) ~", paste(var, collapse = " + ")))
  formula_without_tvf_without_cluster <<- as.formula(paste("Surv(tstop, status) ~",         paste(var, collapse = " + ")))

  tmp <- file.path(tempdir(), "tvf_cox"); create_output_folder(tmp)
  sp  <- split_data(df, train_prop = 0.75, seed = 8888)
  res <- coxph_func(df, tmp, tvf = TRUE, sp)

  expect_true(!is.null(res$fit))
  expect_true(is.numeric(res$concord))
})
