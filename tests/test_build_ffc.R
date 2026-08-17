#!/usr/bin/env Rscript

# Focused, dependency-light tests for the FFC analysis.  These deliberately do
# not load the private FFC inputs or run Amelia.

source(file.path("src", "analysis", "build_ffc_seeds.R"))

expect_true <- function(value, message = "expect_true failed") {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

expect_equal <- function(actual, expected, tolerance = 1e-12,
                         message = "expect_equal failed") {
  if (!isTRUE(all.equal(actual, expected, tolerance = tolerance))) {
    stop(sprintf(
      "%s\nactual: %s\nexpected: %s",
      message, paste(actual, collapse = ", "),
      paste(expected, collapse = ", ")
    ), call. = FALSE)
  }
}

expect_error <- function(expression, message = "expected an error") {
  errored <- FALSE
  tryCatch(force(expression), error = function(error) errored <<- TRUE)
  if (!errored) stop(message, call. = FALSE)
}

# Defaults implement the requested 100 x 100 crossed design and independent
# 1,000-run observed-data sweep.
settings <- ffc_settings()
expect_equal(settings$n_bootstrap_resamples, 100L)
expect_equal(settings$n_seed_vectors, 100L)
expect_equal(settings$n_visualization_runs, 1000L)
expect_equal(settings$n_jobs, -1L)
expect_equal(settings$bootstrap_batch_size, 0L)
expect_equal(resolve_worker_count(-1L), parallel::detectCores(logical = TRUE))
expect_equal(resolve_batch_size(100L, 0L, -1L),
             min(100L, parallel::detectCores(logical = TRUE)))
expect_equal(settings$fixed_seed, 8544L)

# Imputation seeds form one shared stage-major block large enough for both
# experiments; bootstrap seeds occupy the following, disjoint block.
plan <- make_ffc_seed_plan(0:1099, settings)
expect_equal(length(plan$imputation), 1000L)
expect_equal(length(plan$bootstrap), 100L)
expect_equal(plan$imputation, 0:999)
expect_equal(plan$bootstrap, 1000:1099)
expect_true(length(intersect(plan$imputation, plan$bootstrap)) == 0L,
            "imputation and bootstrap seed blocks overlap")
expect_error(make_ffc_seed_plan(0:1098, settings),
             "an undersized seed list should be rejected")

# Six outcomes map to nine valid model pairs and two metrics per pair.
expected_pairs <- data.frame(
  outcome = c(
    "gpa", "grit", "materialHardship",
    rep("eviction", 2L), rep("layoff", 2L), rep("jobTraining", 2L)
  ),
  account = c(rep("ols", 3L), rep(c("ols", "logit"), 3L)),
  stringsAsFactors = FALSE
)
expect_equal(.ffc_valid_pairs, expected_pairs)
expect_equal(length(FFC_ESTIMANDS), 18L)
expect_true(!anyDuplicated(FFC_ESTIMANDS), "estimands must be unique")
expected_estimands <- unlist(lapply(seq_len(nrow(expected_pairs)), function(i) {
  paste(expected_pairs$outcome[[i]], expected_pairs$account[[i]],
        c("R2", "beta"), sep = "_")
}), use.names = FALSE)
expect_equal(FFC_ESTIMANDS, expected_estimands)
expect_equal(
  unname(FFC_PRIMARY_ACCOUNT[FFC_OUTCOMES]),
  c("ols", "ols", "ols", "logit", "logit", "logit")
)

# Flattening an outcome/model result table preserves the exact estimand names
# and maps both metrics to the intended positions.
score_rows <- expected_pairs
score_rows$r2_holdout <- seq_len(nrow(score_rows)) / 10
score_rows$beta <- seq_len(nrow(score_rows)) + 0.25
scores <- .rows_to_score_vector(score_rows)
expect_equal(names(scores), FFC_ESTIMANDS)
expected_scores <- setNames(numeric(length(FFC_ESTIMANDS)), FFC_ESTIMANDS)
for (i in seq_len(nrow(score_rows))) {
  prefix <- paste(score_rows$outcome[[i]], score_rows$account[[i]], sep = "_")
  expected_scores[[paste0(prefix, "_R2")]] <- score_rows$r2_holdout[[i]]
  expected_scores[[paste0(prefix, "_beta")]] <- score_rows$beta[[i]]
}
expect_equal(scores, expected_scores)
expect_error(.rows_to_score_vector(score_rows[-1L, ]),
             "a missing valid outcome/model pair should be rejected")

# Stratified resampling preserves partition sizes and never draws a row from a
# different split into a position; resampling also refreshes Amelia's row ID.
panel <- data.frame(
  row_id = 10:15,
  split = c("train", "train", "train", "test", "test", "other"),
  marker = letters[1:6],
  stringsAsFactors = FALSE
)
indices <- stratified_bootstrap_index_block(panel$split, c(41L, 73L, 109L))
expect_equal(dim(indices), c(3L, nrow(panel)))
for (i in seq_len(nrow(indices))) {
  expect_equal(panel$split[indices[i, ]], panel$split)
  sampled <- resample_ffc_panel(panel, indices[i, ])
  expect_equal(sampled$split, panel$split)
  expect_equal(sampled$row_id, seq_len(nrow(panel)))
}

# Small deterministic helpers retain the published score and bounding rules.
expect_equal(
  .holdout_pseudo_r2(c(1, 3), c(1, 2), training_mean = 2, context = "test"),
  0.5
)
expect_equal(.clamp_prediction(c(0, 2, 5), "gpa"), c(1, 2, 4))
expect_equal(.clamp_prediction(c(-1, 0.4, 2), "eviction"), c(0, 0.4, 1))
expect_equal(.positive_or_na(c(-1, 0, 2, NA)), c(NA, NA, 2, NA))
expect_equal(.indicator_or_na(c(-1, 0, 1, 2, NA)), c(NA, NA, 1, 0, NA))
expect_error(.holdout_pseudo_r2(c(1, 1), c(1, 1), 1, "constant truth"))

cat("build_ffc_seeds.R tests passed\n")
