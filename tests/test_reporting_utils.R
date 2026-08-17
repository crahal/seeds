#!/usr/bin/env Rscript

source(file.path("src", "analysis", "reporting_utils.R"))

expect_true <- function(value, message = "expect_true failed") {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

expect_equal <- function(actual, expected, tolerance = 1e-12,
                         message = "expect_equal failed") {
  if (!isTRUE(all.equal(actual, expected, tolerance = tolerance,
                        check.attributes = FALSE))) {
    stop(sprintf("%s\nactual: %s\nexpected: %s", message,
                 paste(actual, collapse = ", "),
                 paste(expected, collapse = ", ")), call. = FALSE)
  }
}

expect_error <- function(expression, message = "expected an error") {
  errored <- FALSE
  tryCatch(force(expression), error = function(error) errored <<- TRUE)
  if (!errored) stop(message, call. = FALSE)
}

# Known additive random-effects grid.
data_effect <- c(-1, 0, 1, 2)
seed_effect <- c(0, 2, 4)
scores <- outer(data_effect, seed_effect, "+")
diagnostics <- crossed_s5_diagnostics(scores)
expect_equal(diagnostics$seed_averaged_estimate, mean(scores))
expect_equal(diagnostics$data_variance, var(data_effect))
expect_equal(diagnostics$between_seed_variance, var(seed_effect))
expect_equal(diagnostics$data_seed_interaction_variance, 0)
expected_share <- var(seed_effect) / (var(seed_effect) + var(data_effect))
expect_equal(diagnostics$algorithmic_variance_share, expected_share)
expect_equal(diagnostics$total_order_algorithmic_variance_share,
             expected_share)

# Row/column order cannot affect any variance component.
arbitrary <- matrix(c(
  0.2, 0.8, 0.4,
  0.7, 0.1, 0.5,
  0.3, 0.9, 0.6,
  0.4, 0.2, 1.0
), nrow = 4L, byrow = TRUE)
expected <- crossed_s5_diagnostics(arbitrary)
actual <- crossed_s5_diagnostics(arbitrary[c(3, 1, 4, 2), c(2, 3, 1)])
for (field in c("data_variance", "between_seed_variance",
                "data_main_effect_variance",
                "data_seed_interaction_variance")) {
  expect_equal(actual[[field]], expected[[field]])
}

# The raw negative component remains auditable while reported variance is zero.
negative_seed <- crossed_s5_diagnostics(matrix(c(1, -1, -1, 1), nrow = 2L,
                                                byrow = TRUE))
expect_true(negative_seed$between_seed_variance_raw < 0)
expect_equal(negative_seed$between_seed_variance, 0)
expect_true(negative_seed$variance_component_boundary_hit)
expect_equal(negative_seed$between_seed_variability_sd, 0)
expect_equal(negative_seed$relative_importance, 0)
expect_equal(negative_seed$algorithmic_variance_share, 0)
report <- format_s5_report(negative_seed, "test")
expect_true(grepl("sigma_S_squared_adj_raw=-2", report, fixed = TRUE))
expect_true(grepl("does not prove", report, fixed = TRUE))

# A constant grid has undefined 0/0 ratios and shares.
constant <- crossed_s5_diagnostics(matrix(1, nrow = 3L, ncol = 4L))
expect_equal(constant$data_uncertainty_sd, 0)
expect_equal(constant$between_seed_variability_sd, 0)
expect_true(is.na(constant$relative_importance))
expect_true(is.na(constant$algorithmic_variance_share))
expect_true(is.na(constant$total_order_algorithmic_variance_share))

# Negative finite-sample V_D invalidates only the total-order share.
interaction <- matrix(c(
  1, -1, 0,
  -1, 1, 0,
  0, 0, 0
), nrow = 3L, byrow = TRUE)
negative_data <- crossed_s5_diagnostics(
  sweep(interaction, 2L, c(-2, 0, 2), "+")
)
expect_true(negative_data$between_seed_variance >= 0)
expect_true(negative_data$data_main_effect_variance < 0)
expect_true(is.finite(negative_data$algorithmic_variance_share))
expect_true(is.na(negative_data$total_order_algorithmic_variance_share))

# Invalid grids are rejected.
expect_error(crossed_s5_diagnostics(c(1, 2, 3)))
expect_error(crossed_s5_diagnostics(matrix(1, nrow = 1L, ncol = 3L)))
bad <- matrix(c(1, NA, 2, 3), nrow = 2L)
expect_error(crossed_s5_diagnostics(bad))

# Stage-major component allocation uses a zero-based offset.
blocks <- seed_component_blocks(0:19, 3L, c("folding", "modeling"), offset = 2L)
expect_equal(blocks$folding, 2:4)
expect_equal(blocks$modeling, 5:7)

# Parallel controls resolve portable all-core syntax and automatic batches.
expect_true(resolve_worker_count(-1L) >= 1L)
expect_equal(resolve_worker_count(3L, n_tasks = 2L), 2L)
expect_equal(resolve_batch_size(10L, batch_size = 0L, n_jobs = 3L), 3L)
expect_equal(resolve_batch_size(10L, batch_size = 4L, n_jobs = 3L), 4L)
expect_error(resolve_worker_count(0L))

# Bootstrap blocks are reproducible and do not alter caller RNG state.
first <- bootstrap_index_block(7L, c(11L, 22L))
second <- bootstrap_index_block(7L, c(11L, 22L))
expect_equal(first, second)
expect_equal(dim(first), c(2L, 7L))
set.seed(99)
expected_random <- runif(3)
set.seed(99)
invisible(bootstrap_index_block(5L, c(4L, 8L)))
actual_random <- runif(3)
expect_equal(actual_random, expected_random)

# Stratified rows retain each stratum's positions and counts.
strata <- c("train", "train", "test", "test", "other")
stratified <- stratified_bootstrap_index_block(strata, c(101L, 202L))
expect_equal(dim(stratified), c(2L, 5L))
expect_true(all(stratified[, 1:2] %in% 1:2))
expect_true(all(stratified[, 3:4] %in% 3:4))
expect_true(all(stratified[, 5] == 5L))

summary <- observed_seed_summary(c(1, 2, 3))
expect_equal(summary$observed_data_seed_average, 2)
expect_equal(summary$observed_between_seed_sd, 1)

for (heading in paste0(1:6, ".")) {
  expect_true(grepl(heading, report, fixed = TRUE))
}

log_path <- tempfile(fileext = ".log")
writeLines(report, log_path)
validate_s5_log(log_path, "test")
writeLines(sub("6. Computational details", "", report, fixed = TRUE), log_path)
expect_error(validate_s5_log(log_path, "test"))
unlink(log_path)

cat("reporting_utils.R tests passed\n")
