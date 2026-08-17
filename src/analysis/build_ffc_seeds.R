#!/usr/bin/env Rscript

# FFC seed and sampling-uncertainty analysis.
#
# The diagnostic experiment crosses shared external bootstrap datasets with
# algorithmic seed vectors.  A separate observed-data seed sweep supplies the
# plot input and is never mixed into the S5 variance-component estimates.

options(stringsAsFactors = FALSE)

.ffc_script_path <- function() {
  # When this file is sourced, the innermost source frame identifies this
  # file; commandArgs() instead identifies the outer Rscript entry point.
  frames <- sys.frames()
  for (index in rev(seq_along(frames))) {
    candidate <- frames[[index]]$ofile
    if (!is.null(candidate) && nzchar(candidate)) {
      return(normalizePath(candidate, mustWork = TRUE))
    }
  }
  arguments <- commandArgs(trailingOnly = FALSE)
  file_argument <- arguments[startsWith(arguments, "--file=")]
  if (length(file_argument)) {
    candidate <- sub("^--file=", "", file_argument[[1L]])
    if (candidate != "-" && file.exists(candidate)) {
      return(normalizePath(candidate, mustWork = TRUE))
    }
  }
  normalizePath(file.path("src", "analysis", "build_ffc_seeds.R"),
                mustWork = TRUE)
}

FFC_SCRIPT_PATH <- .ffc_script_path()
FFC_ANALYSIS_DIR <- dirname(FFC_SCRIPT_PATH)
FFC_REPO_ROOT <- normalizePath(file.path(FFC_ANALYSIS_DIR, "..", ".."),
                               mustWork = TRUE)
source(file.path(FFC_ANALYSIS_DIR, "reporting_utils.R"))

FFC_OUTCOMES <- c(
  "gpa", "grit", "materialHardship", "eviction", "layoff", "jobTraining"
)
FFC_BINARY_OUTCOMES <- c("eviction", "layoff", "jobTraining")
FFC_CONTINUOUS_OUTCOMES <- setdiff(FFC_OUTCOMES, FFC_BINARY_OUTCOMES)
FFC_ACCOUNTS <- c("ols", "logit")
FFC_PRIMARY_ACCOUNT <- c(
  gpa = "ols",
  grit = "ols",
  materialHardship = "ols",
  eviction = "logit",
  layoff = "logit",
  jobTraining = "logit"
)
FFC_ISSUE_NAMES <- c(
  "amelia_warnings", "ols_fit_warnings", "logit_fit_warnings",
  "fit_fallbacks", "metric_substitutions"
)

.ffc_valid_pairs <- do.call(rbind, lapply(FFC_OUTCOMES, function(outcome) {
  accounts <- if (outcome %in% FFC_BINARY_OUTCOMES) FFC_ACCOUNTS else "ols"
  data.frame(outcome = outcome, account = accounts, stringsAsFactors = FALSE)
}))
FFC_ESTIMANDS <- unlist(lapply(seq_len(nrow(.ffc_valid_pairs)), function(index) {
  pair <- .ffc_valid_pairs[index, ]
  paste(pair$outcome, pair$account, c("R2", "beta"), sep = "_")
}), use.names = FALSE)

ffc_settings <- function(n_bootstrap_resamples = 100L,
                         n_seed_vectors = 100L,
                         n_visualization_runs = 1000L,
                         n_jobs = -1L,
                         bootstrap_batch_size = 0L,
                         fixed_seed = 8544L) {
  settings <- list(
    n_bootstrap_resamples = as.integer(n_bootstrap_resamples),
    n_seed_vectors = as.integer(n_seed_vectors),
    n_visualization_runs = as.integer(n_visualization_runs),
    n_jobs = as.integer(n_jobs),
    bootstrap_batch_size = as.integer(bootstrap_batch_size),
    fixed_seed = as.integer(fixed_seed)
  )
  if (any(!is.finite(unlist(settings)))) {
    stop("all settings must be finite", call. = FALSE)
  }
  if (settings$n_bootstrap_resamples < 2L || settings$n_seed_vectors < 2L) {
    stop("S5 diagnostics require at least two bootstraps and seed vectors",
         call. = FALSE)
  }
  if (settings$n_visualization_runs < 2L) {
    stop("n_visualization_runs must be at least two", call. = FALSE)
  }
  resolve_worker_count(settings$n_jobs)
  if (settings$bootstrap_batch_size < 0L) {
    stop("bootstrap_batch_size cannot be negative", call. = FALSE)
  }
  .validate_seeds(settings$fixed_seed, "fixed_seed")
  settings
}

make_ffc_seed_plan <- function(seed_list, settings) {
  n_vectors <- max(settings$n_seed_vectors, settings$n_visualization_runs)
  components <- seed_component_blocks(
    seed_list, n_vectors, "imputation", offset = 0L
  )
  bootstrap_start <- n_vectors
  bootstrap_stop <- bootstrap_start + settings$n_bootstrap_resamples
  if (bootstrap_stop > length(seed_list)) {
    stop(sprintf("seed list has %d entries but %d are required",
                 length(seed_list), bootstrap_stop), call. = FALSE)
  }
  list(
    imputation = components$imputation,
    bootstrap = as.integer(seed_list[(bootstrap_start + 1L):bootstrap_stop])
  )
}

.read_selected_csv <- function(path, selected_columns) {
  header <- names(read.csv(path, nrows = 0L, check.names = FALSE))
  missing <- setdiff(selected_columns, header)
  if (length(missing)) {
    stop(sprintf("%s is missing columns: %s", path,
                 paste(missing, collapse = ", ")), call. = FALSE)
  }
  classes <- rep("NULL", length(header))
  classes[header %in% selected_columns] <- NA
  read.csv(path, colClasses = classes, check.names = FALSE,
           na.strings = c("", "NA"))
}

.positive_or_na <- function(value) {
  ifelse(!is.na(value) & value > 0, value, NA_real_)
}

.nonnegative_reverse_or_na <- function(value) {
  ifelse(!is.na(value) & value >= 0, 4 - value, NA_real_)
}

.indicator_or_na <- function(value) {
  ifelse(!is.na(value) & value > 0, as.numeric(value == 1), NA_real_)
}

.sum_columns_preserving_na <- function(data, columns, transform) {
  values <- lapply(columns, function(column) transform(data[[column]]))
  Reduce(`+`, values)
}

prepare_ffc_data <- function(background_csv, train_csv, test_csv) {
  raw_columns <- c(
    "challengeID", "cm1relf", "cm1ethrace", "cm1edu",
    "t5c13a", "t5c13b", "t5c13c", "t5b2b", "t5b4y", "t5b4z",
    "m5a2", paste0("m5f23", letters[1:10]),
    "f5a2", paste0("f5f23", letters[1:10]),
    paste0("n5g1", letters[1:10]), "m5i4", "f5i4", "m5i3b", "f5i3b"
  )
  background <- .read_selected_csv(background_csv, raw_columns)
  train <- read.csv(train_csv, check.names = FALSE, na.strings = c("", "NA"))
  test <- read.csv(test_csv, check.names = FALSE, na.strings = c("", "NA"))
  required_outcome_columns <- c("challengeID", FFC_OUTCOMES)
  for (input in list(train = train, test = test)) {
    missing <- setdiff(required_outcome_columns, names(input))
    if (length(missing)) {
      stop(sprintf("FFC outcome data are missing: %s",
                   paste(missing, collapse = ", ")), call. = FALSE)
    }
  }
  if (anyDuplicated(background$challengeID) || anyDuplicated(train$challengeID) ||
      anyDuplicated(test$challengeID)) {
    stop("challengeID values must be unique within each input", call. = FALSE)
  }
  if (length(intersect(train$challengeID, test$challengeID))) {
    stop("train and test challengeID values must be disjoint", call. = FALSE)
  }
  if (any(!train$challengeID %in% background$challengeID) ||
      any(!test$challengeID %in% background$challengeID)) {
    stop("every train/test challengeID must occur in background", call. = FALSE)
  }

  race <- ifelse(
    background$cm1ethrace %in% c(1, 4), "White/other",
    ifelse(background$cm1ethrace == 2, "Black",
           ifelse(background$cm1ethrace == 3, "Hispanic", NA_character_))
  )
  relationship <- ifelse(
    background$cm1relf == 1, "Married",
    ifelse(background$cm1relf == 2, "Cohabiting",
           ifelse(background$cm1relf >= 3, "Other", NA_character_))
  )
  education_code <- ifelse(background$cm1edu >= 1, background$cm1edu, NA)
  education <- factor(
    education_code, levels = 1:4,
    labels = c("Less than high school", "High school", "Some college", "College")
  )
  gpa9 <- (
    .positive_or_na(background$t5c13a) +
      .positive_or_na(background$t5c13b) +
      .positive_or_na(background$t5c13c)
  ) / 3
  grit9 <- (
    .positive_or_na(background$t5b2b) +
      .nonnegative_reverse_or_na(background$t5b4y) +
      .nonnegative_reverse_or_na(background$t5b4z)
  ) / 3
  mother_hardship <- .sum_columns_preserving_na(
    background, paste0("m5f23", letters[1:10]), .indicator_or_na
  ) / 10
  father_hardship <- .sum_columns_preserving_na(
    background, paste0("f5f23", letters[1:10]), .indicator_or_na
  ) / 10
  other_hardship <- .sum_columns_preserving_na(
    background, paste0("n5g1", letters[1:10]), .indicator_or_na
  ) / 10
  mother_available <- background$m5a2 %in% c(1, 2)
  father_available <- background$f5a2 %in% c(1, 2)
  material_hardship9 <- ifelse(
    mother_available, mother_hardship,
    ifelse(father_available, father_hardship, other_hardship)
  )
  eviction9 <- ifelse(
    mother_available,
    ifelse(background$m5f23d <= 0, NA, background$m5f23d == 1),
    ifelse(
      father_available,
      ifelse(background$f5f23d <= 0, NA, background$f5f23d == 1),
      NA
    )
  )
  layoff9 <- ifelse(
    mother_available,
    ifelse(background$m5i4 > 0, background$m5i4 == 2, NA),
    ifelse(
      father_available,
      ifelse(background$f5i4 > 0, background$f5i4 == 2, NA),
      NA
    )
  )
  job_training9 <- ifelse(
    mother_available,
    ifelse(background$m5i3b > 0, background$m5i3b == 1, NA),
    ifelse(
      father_available,
      ifelse(background$f5i3b > 0, background$f5i3b == 1, NA),
      NA
    )
  )

  panel <- data.frame(
    challengeID = background$challengeID,
    cm1ethrace = race,
    cm1relf = relationship,
    cm1edu = education,
    gpa9 = gpa9,
    grit9 = grit9,
    materialHardship9 = material_hardship9,
    eviction9 = as.numeric(eviction9),
    layoff9 = as.numeric(layoff9),
    jobTraining9 = as.numeric(job_training9),
    stringsAsFactors = FALSE
  )
  train_positions <- match(panel$challengeID, train$challengeID)
  test_positions <- match(panel$challengeID, test$challengeID)
  for (outcome in FFC_OUTCOMES) {
    panel[[outcome]] <- train[[outcome]][train_positions]
    panel[[paste0("truth_", outcome)]] <- test[[outcome]][test_positions]
  }
  panel$split <- ifelse(
    !is.na(train_positions), "train",
    ifelse(!is.na(test_positions), "test", "other")
  )
  expected_split_counts <- c(train = nrow(train), test = nrow(test),
                             other = nrow(background) - nrow(train) - nrow(test))
  actual_split_counts <- table(factor(panel$split, levels = names(expected_split_counts)))
  if (!identical(as.integer(actual_split_counts),
                 as.integer(expected_split_counts))) {
    stop("unexpected train/test/other split counts", call. = FALSE)
  }
  missing_check_columns <- c(
    "cm1ethrace", "cm1relf", "cm1edu", paste0(FFC_OUTCOMES, "9"),
    FFC_OUTCOMES
  )
  all_missing <- apply(panel[, missing_check_columns, drop = FALSE], 1L,
                       function(row) all(is.na(row)))
  panel$cm1ethrace[all_missing] <- "White/other"
  panel$cm1ethrace <- factor(
    panel$cm1ethrace, levels = c("White/other", "Black", "Hispanic")
  )
  panel$cm1relf <- factor(
    panel$cm1relf, levels = c("Married", "Cohabiting", "Other")
  )
  panel$row_id <- seq_len(nrow(panel))
  panel
}

.capture_warnings <- function(expression) {
  warning_messages <- character()
  value <- withCallingHandlers(
    expression,
    warning = function(condition) {
      warning_messages <<- c(warning_messages, conditionMessage(condition))
      invokeRestart("muffleWarning")
    }
  )
  list(value = value, warnings = warning_messages)
}

.fit_with_audit <- function(expression, context) {
  tryCatch(
    .capture_warnings(expression),
    error = function(error) {
      stop(sprintf("%s failed: %s", context, conditionMessage(error)),
           call. = FALSE)
    }
  )
}

.holdout_pseudo_r2 <- function(truth, prediction, training_mean, context) {
  truth <- as.numeric(truth)
  prediction <- as.numeric(prediction)
  valid_prediction <- is.finite(truth) & is.finite(prediction)
  valid_truth <- is.finite(truth)
  if (!any(valid_prediction) || !any(valid_truth) || !is.finite(training_mean)) {
    stop(sprintf("%s has no finite holdout observations", context),
         call. = FALSE)
  }
  numerator <- mean((truth[valid_prediction] - prediction[valid_prediction])^2)
  denominator <- mean((truth[valid_truth] - training_mean)^2)
  score <- 1 - numerator / denominator
  if (!is.finite(score) || denominator <= 0) {
    stop(sprintf("%s produced a nonfinite pseudo-R2", context), call. = FALSE)
  }
  score
}

.lag_column <- function(outcome) paste0(outcome, "9")

.clamp_prediction <- function(prediction, outcome) {
  if (outcome %in% c("gpa", "grit")) {
    pmin(4, pmax(1, prediction))
  } else {
    pmin(1, pmax(0, prediction))
  }
}

impute_ffc_outcome <- function(panel, outcome) {
  lag_name <- .lag_column(outcome)
  imputation_data <- panel[, c(
    "row_id", "cm1ethrace", "cm1relf", "cm1edu", lag_name, outcome
  ), drop = FALSE]
  audited <- tryCatch(
    .capture_warnings(
      Amelia::amelia(
        imputation_data,
        m = 1L,
        p2s = 0L,
        noms = c("cm1ethrace", "cm1relf"),
        ords = "cm1edu",
        idvars = "row_id"
      )
    ),
    error = function(error) {
      stop(sprintf("Amelia imputation failed for %s: %s", outcome,
                   conditionMessage(error)), call. = FALSE)
    }
  )
  imputed <- audited$value$imputations[[1L]]
  if (nrow(imputed) != nrow(panel) || any(imputed$row_id != panel$row_id)) {
    stop(sprintf("Amelia changed row identity/order for %s", outcome),
         call. = FALSE)
  }
  if (any(!is.finite(imputed[[lag_name]])) ||
      any(!is.finite(imputed[[outcome]]))) {
    stop(sprintf("Amelia left incomplete numeric values for %s", outcome),
         call. = FALSE)
  }
  list(data = imputed, warnings = audited$warnings)
}

evaluate_ffc_outcome <- function(panel, outcome) {
  imputation <- impute_ffc_outcome(panel, outcome)
  imputed <- imputation$data
  lag_name <- .lag_column(outcome)
  formula <- reformulate(
    c("cm1ethrace", "cm1relf", "cm1edu", lag_name), response = outcome
  )
  training_rows <- panel$split == "train" & !is.na(panel[[outcome]])
  test_rows <- panel$split == "test"
  if (sum(training_rows) < 2L || !any(test_rows)) {
    stop(sprintf("insufficient train/test rows for %s", outcome), call. = FALSE)
  }
  training_mean <- mean(panel[[outcome]][training_rows], na.rm = TRUE)
  truth <- panel[[paste0("truth_", outcome)]][test_rows]
  issues <- setNames(integer(length(FFC_ISSUE_NAMES)), FFC_ISSUE_NAMES)
  issues[["amelia_warnings"]] <- length(imputation$warnings)

  ols_audit <- .fit_with_audit(
    lm(formula, data = imputed[training_rows, , drop = FALSE]),
    sprintf("OLS fit for %s", outcome)
  )
  issues[["ols_fit_warnings"]] <- length(ols_audit$warnings)
  ols <- ols_audit$value
  ols_prediction <- tryCatch(
    predict(ols, newdata = imputed[test_rows, , drop = FALSE]),
    error = function(error) {
      stop(sprintf("OLS prediction failed for %s: %s", outcome,
                   conditionMessage(error)), call. = FALSE)
    }
  )
  ols_prediction <- .clamp_prediction(ols_prediction, outcome)
  ols_beta <- unname(coef(ols)[[lag_name]])
  if (!is.finite(ols_beta) || any(!is.finite(ols_prediction))) {
    stop(sprintf("OLS returned nonfinite values for %s", outcome), call. = FALSE)
  }
  ols_r2 <- .holdout_pseudo_r2(
    truth, ols_prediction, training_mean, sprintf("%s/OLS", outcome)
  )

  rows <- data.frame(
    outcome = rep(outcome, 2L),
    account = FFC_ACCOUNTS,
    r2_holdout = c(ols_r2, NA_real_),
    beta = c(ols_beta, NA_real_),
    stringsAsFactors = FALSE
  )
  if (outcome %in% FFC_BINARY_OUTCOMES) {
    observed_classes <- unique(panel[[outcome]][training_rows])
    if (length(observed_classes) != 2L) {
      stop(sprintf("bootstrap training data for %s do not contain both classes",
                   outcome), call. = FALSE)
    }
    logit_audit <- .fit_with_audit(
      glm(formula, family = binomial(link = "logit"),
          data = imputed[training_rows, , drop = FALSE]),
      sprintf("Logit fit for %s", outcome)
    )
    issues[["logit_fit_warnings"]] <- length(logit_audit$warnings)
    logit <- logit_audit$value
    logit_prediction <- tryCatch(
      predict(logit, newdata = imputed[test_rows, , drop = FALSE],
              type = "response"),
      error = function(error) {
        stop(sprintf("Logit prediction failed for %s: %s", outcome,
                     conditionMessage(error)), call. = FALSE)
      }
    )
    logit_prediction <- pmin(1, pmax(0, logit_prediction))
    logit_beta <- unname(coef(logit)[[lag_name]])
    if (!is.finite(logit_beta) || any(!is.finite(logit_prediction))) {
      stop(sprintf("Logit returned nonfinite values for %s", outcome),
           call. = FALSE)
    }
    rows$r2_holdout[rows$account == "logit"] <- .holdout_pseudo_r2(
      truth, logit_prediction, training_mean, sprintf("%s/Logit", outcome)
    )
    rows$beta[rows$account == "logit"] <- logit_beta
  }
  list(rows = rows, issues = issues)
}

.rows_to_score_vector <- function(rows) {
  scores <- setNames(rep(NA_real_, length(FFC_ESTIMANDS)), FFC_ESTIMANDS)
  for (index in seq_len(nrow(rows))) {
    if (!is.finite(rows$r2_holdout[[index]])) next
    prefix <- paste(rows$outcome[[index]], rows$account[[index]], sep = "_")
    scores[[paste0(prefix, "_R2")]] <- rows$r2_holdout[[index]]
    scores[[paste0(prefix, "_beta")]] <- rows$beta[[index]]
  }
  if (any(!is.finite(scores))) {
    missing <- names(scores)[!is.finite(scores)]
    stop(sprintf("FFC evaluation did not produce estimands: %s",
                 paste(missing, collapse = ", ")), call. = FALSE)
  }
  scores
}

evaluate_ffc_seed <- function(panel, imputation_seed) {
  imputation_seed <- .validate_seeds(imputation_seed, "imputation_seed")
  set.seed(imputation_seed)
  outcome_results <- lapply(
    FFC_OUTCOMES, function(outcome) evaluate_ffc_outcome(panel, outcome)
  )
  rows <- do.call(rbind, lapply(outcome_results, `[[`, "rows"))
  rownames(rows) <- NULL
  issues <- Reduce(`+`, lapply(outcome_results, `[[`, "issues"))
  list(rows = rows, scores = .rows_to_score_vector(rows), issues = issues)
}

resample_ffc_panel <- function(panel, indices) {
  sampled <- panel[indices, , drop = FALSE]
  rownames(sampled) <- NULL
  sampled$row_id <- seq_len(nrow(sampled))
  sampled
}

.parallel_lapply <- function(values, function_to_apply, n_jobs) {
  cores <- resolve_worker_count(n_jobs, n_tasks = length(values))
  if (cores <= 1L) {
    return(lapply(values, function_to_apply))
  }
  if (.Platform$OS.type == "windows") {
    warning("parallel FFC execution requires fork support; running serially")
    return(lapply(values, function_to_apply))
  }
  results <- parallel::mclapply(
    values, function_to_apply, mc.cores = cores, mc.preschedule = TRUE,
    mc.set.seed = FALSE
  )
  failed <- vapply(results, inherits, logical(1L), what = "try-error")
  if (any(failed)) {
    stop(sprintf("parallel FFC task failed: %s", results[[which(failed)[1L]]]),
         call. = FALSE)
  }
  results
}

.evaluate_bootstrap_row <- function(bootstrap_number, bootstrap_indices,
                                    panel, imputation_seeds) {
  sampled_panel <- resample_ffc_panel(
    panel, bootstrap_indices[bootstrap_number, ]
  )
  score_rows <- matrix(
    NA_real_, nrow = length(imputation_seeds), ncol = length(FFC_ESTIMANDS),
    dimnames = list(NULL, FFC_ESTIMANDS)
  )
  issue_counts <- setNames(integer(length(FFC_ISSUE_NAMES)), FFC_ISSUE_NAMES)
  for (seed_number in seq_along(imputation_seeds)) {
    result <- evaluate_ffc_seed(sampled_panel, imputation_seeds[[seed_number]])
    score_rows[seed_number, ] <- result$scores
    issue_counts <- issue_counts + result$issues
  }
  list(
    bootstrap_number = bootstrap_number,
    scores = score_rows,
    issues = issue_counts
  )
}

run_crossed_grid <- function(panel, bootstrap_indices, seed_plan, settings,
                             checkpoint_dir, logger) {
  B <- settings$n_bootstrap_resamples
  m <- settings$n_seed_vectors
  if (!identical(dim(bootstrap_indices), c(B, nrow(panel)))) {
    stop(sprintf(
      "bootstrap index block has shape %s; expected %dx%d",
      paste(dim(bootstrap_indices), collapse = "x"), B, nrow(panel)
    ), call. = FALSE)
  }
  matrices <- setNames(lapply(FFC_ESTIMANDS, function(unused) {
    matrix(NA_real_, nrow = B, ncol = m)
  }), FFC_ESTIMANDS)
  issue_counts <- setNames(integer(length(FFC_ISSUE_NAMES)), FFC_ISSUE_NAMES)
  imputation_seeds <- seed_plan$imputation[seq_len(m)]
  batch_size <- resolve_batch_size(
    B, settings$bootstrap_batch_size, settings$n_jobs
  )
  dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)
  started <- proc.time()[["elapsed"]]
  for (start in seq.int(1L, B, by = batch_size)) {
    stop_number <- min(start + batch_size - 1L, B)
    bootstrap_numbers <- start:stop_number
    results <- .parallel_lapply(
      bootstrap_numbers,
      function(number) .evaluate_bootstrap_row(
        number, bootstrap_indices, panel, imputation_seeds
      ),
      settings$n_jobs
    )
    for (result in results) {
      number <- result$bootstrap_number
      for (estimand in FFC_ESTIMANDS) {
        matrices[[estimand]][number, ] <- result$scores[, estimand]
      }
      issue_counts <- issue_counts + result$issues
    }
    atomic_save_rds(
      matrices,
      file.path(checkpoint_dir, "ffc_crossed_score_matrices_checkpoint.rds")
    )
    atomic_save_rds(
      issue_counts,
      file.path(checkpoint_dir, "ffc_crossed_issue_counts_checkpoint.rds")
    )
    logger$info(
      "Crossed grid: %d/%d cells complete (%.1f seconds elapsed)",
      stop_number * m, B * m, proc.time()[["elapsed"]] - started
    )
  }
  incomplete <- vapply(
    matrices, function(values) any(!is.finite(values)), logical(1L)
  )
  if (any(incomplete)) {
    stop(sprintf("crossed experiment has incomplete matrices: %s",
                 paste(names(incomplete)[incomplete], collapse = ", ")),
         call. = FALSE)
  }
  list(matrices = matrices, issues = issue_counts)
}

run_observed_seed_sweep <- function(panel, seed_plan, settings, logger) {
  n_runs <- settings$n_visualization_runs
  seeds <- seed_plan$imputation[seq_len(n_runs)]
  started <- proc.time()[["elapsed"]]
  results <- .parallel_lapply(
    seq_len(n_runs),
    function(index) evaluate_ffc_seed(panel, seeds[[index]]),
    settings$n_jobs
  )
  rows <- do.call(rbind, lapply(seq_along(results), function(index) {
    frame <- results[[index]]$rows
    frame$seed <- seeds[[index]]
    frame
  }))
  rownames(rows) <- NULL
  score_matrix <- do.call(rbind, lapply(results, `[[`, "scores"))
  colnames(score_matrix) <- FFC_ESTIMANDS
  issue_counts <- Reduce(`+`, lapply(results, `[[`, "issues"))
  if (any(!is.finite(score_matrix))) {
    stop("observed-data sweep contains incomplete estimands", call. = FALSE)
  }
  logger$info(
    "Observed-data sweep complete: %d seeds, %d outcome/account rows " %+%
      "(%.1f seconds elapsed)",
    n_runs, nrow(rows), proc.time()[["elapsed"]] - started
  )
  list(rows = rows, scores = score_matrix, issues = issue_counts)
}

`%+%` <- function(left, right) paste0(left, right)

.diagnostics_frame <- function(diagnostics) {
  rows <- lapply(names(diagnostics), function(estimand) {
    parts <- strsplit(estimand, "_", fixed = TRUE)[[1L]]
    data.frame(
      Estimand = estimand,
      Outcome = parts[[1L]],
      Model = parts[[2L]],
      Metric = parts[[3L]],
      as.data.frame(diagnostics[[estimand]], optional = TRUE,
                    stringsAsFactors = FALSE),
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
  })
  result <- do.call(rbind, rows)
  rownames(result) <- NULL
  result
}

.issue_frame <- function(crossed, observed, fixed) {
  data.frame(
    Issue = FFC_ISSUE_NAMES,
    Crossed = as.integer(crossed[FFC_ISSUE_NAMES]),
    Observed = as.integer(observed[FFC_ISSUE_NAMES]),
    Fixed_Seed = as.integer(fixed[FFC_ISSUE_NAMES]),
    stringsAsFactors = FALSE
  )
}

write_ffc_outputs <- function(output_dir, bootstrap_indices, seed_plan,
                              crossed, observed, fixed_seed_result,
                              diagnostics, metadata, provenance, settings) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  B <- settings$n_bootstrap_resamples
  m <- settings$n_seed_vectors
  bootstrap_number <- rep(0:(B - 1L), each = m)
  seed_number <- rep(0:(m - 1L), times = B)
  crossed_frame <- data.frame(
    Bootstrap_Replicate = bootstrap_number,
    Seed_Vector = seed_number,
    Bootstrap_Seed = seed_plan$bootstrap[bootstrap_number + 1L],
    Imputation_Seed = seed_plan$imputation[seed_number + 1L],
    check.names = FALSE
  )
  for (estimand in FFC_ESTIMANDS) {
    crossed_frame[[estimand]] <- as.vector(t(crossed$matrices[[estimand]]))
  }
  write.csv(
    crossed_frame,
    file.path(output_dir, "ffc_crossed_bootstrap_scores.csv"),
    row.names = FALSE, na = "NA"
  )

  visualization_frame <- observed$rows[, c(
    "outcome", "account", "r2_holdout", "beta", "seed"
  )]
  visualization_path <- file.path(output_dir, "ffc_visualization_runs.csv")
  write.csv(visualization_frame, visualization_path,
            row.names = FALSE, na = "NA")
  # The notebook historically hard-codes this filename.  It is now an alias
  # for the requested 1,000-run sweep rather than a claim about row count.
  visualization_aliases <- unique(c(
    file.path(output_dir, sprintf("seed_analysis_%d.csv",
                                  settings$n_visualization_runs)),
    file.path(output_dir, "seed_analysis_10000.csv")
  ))
  copied <- vapply(visualization_aliases, function(path) {
    file.copy(visualization_path, path, overwrite = TRUE)
  }, logical(1L))
  if (any(!copied)) {
    stop("could not write one or more visualization compatibility aliases",
         call. = FALSE)
  }
  fixed_frame <- fixed_seed_result$rows
  fixed_frame$seed <- settings$fixed_seed
  write.csv(
    fixed_frame,
    file.path(output_dir, sprintf("ffc_fixed_seed_%d.csv", settings$fixed_seed)),
    row.names = FALSE, na = "NA"
  )

  write.csv(data.frame(
    Seed_Vector = 0:(m - 1L),
    Imputation_Seed = seed_plan$imputation[seq_len(m)]
  ), file.path(output_dir, "ffc_s5_seed_plan.csv"), row.names = FALSE)
  write.csv(data.frame(
    Seed_Vector = 0:(settings$n_visualization_runs - 1L),
    Imputation_Seed = seed_plan$imputation[
      seq_len(settings$n_visualization_runs)
    ]
  ), file.path(output_dir, "ffc_visualization_seed_plan.csv"), row.names = FALSE)
  write.csv(data.frame(
    Bootstrap_Replicate = 0:(B - 1L),
    Bootstrap_Seed = seed_plan$bootstrap
  ), file.path(output_dir, "ffc_bootstrap_plan.csv"), row.names = FALSE)

  saveRDS(bootstrap_indices,
          file.path(output_dir, "ffc_bootstrap_indices.rds"), compress = FALSE)
  saveRDS(crossed$matrices,
          file.path(output_dir, "ffc_crossed_score_matrices.rds"),
          compress = FALSE)
  for (estimand in FFC_ESTIMANDS) {
    saveRDS(
      crossed$matrices[[estimand]],
      file.path(output_dir, paste0("ffc_crossed_", estimand, ".rds")),
      compress = FALSE
    )
  }
  saveRDS(crossed$issues,
          file.path(output_dir, "ffc_crossed_issue_counts.rds"))
  saveRDS(provenance,
          file.path(output_dir, "ffc_crossed_provenance.rds"))
  saveRDS(metadata, file.path(output_dir, "ffc_run_metadata.rds"))

  diagnostic_frame <- .diagnostics_frame(diagnostics)
  write.csv(
    diagnostic_frame,
    file.path(output_dir, "ffc_s5_diagnostics.csv"),
    row.names = FALSE, na = "NA"
  )
  issues <- .issue_frame(
    crossed$issues, observed$issues, fixed_seed_result$issues
  )
  write.csv(issues, file.path(output_dir, "ffc_issue_counts.csv"),
            row.names = FALSE)
  observed_summaries <- setNames(lapply(FFC_ESTIMANDS, function(estimand) {
    observed_seed_summary(observed$scores[, estimand])
  }), FFC_ESTIMANDS)
  payload <- list(
    metadata = metadata,
    diagnostics = diagnostics,
    crossed_issue_counts = as.list(crossed$issues),
    observed_issue_counts = as.list(observed$issues),
    fixed_seed_issue_counts = as.list(fixed_seed_result$issues),
    observed_data_seed_summaries = observed_summaries
  )
  write_strict_json(
    payload, file.path(output_dir, "ffc_s5_diagnostics.json")
  )
  invisible(list(
    crossed_frame = crossed_frame,
    visualization_frame = visualization_frame,
    diagnostics_frame = diagnostic_frame,
    issue_frame = issues
  ))
}

ffc_usage <- function() {
  cat(paste0(
    "Usage: Rscript src/analysis/build_ffc_seeds.R [options]\n\n",
    "Options:\n",
    "  --background-csv PATH\n",
    "  --train-csv PATH\n",
    "  --test-csv PATH\n",
    "  --seed-list PATH\n",
    "  --output-dir PATH\n",
    "  --log-path PATH\n",
    "  --r-library PATH\n",
    "  --n-bootstrap-resamples N   (default: 100)\n",
    "  --n-seed-vectors N          (default: 100)\n",
    "  --n-visualization-runs N     (default: 1000)\n",
    "  --n-jobs N                   (-1 uses all CPUs; default: -1)\n",
    "  --bootstrap-batch-size N     (0 uses one worker wave; default: 0)\n",
    "  --fixed-seed N               (default: 8544)\n",
    "  --reuse-crossed-scores\n",
    "  --help\n"
  ))
}

parse_ffc_args <- function(arguments) {
  defaults <- list(
    background_csv = file.path(FFC_REPO_ROOT, "data", "ffc", "private",
                               "background.csv"),
    train_csv = file.path(FFC_REPO_ROOT, "data", "ffc", "private", "train.csv"),
    test_csv = file.path(FFC_REPO_ROOT, "data", "ffc", "private", "test.csv"),
    seed_list = file.path(FFC_REPO_ROOT, "assets", "seed_list.txt"),
    output_dir = file.path(FFC_REPO_ROOT, "data", "ffc", "output", "seed"),
    log_path = file.path(FFC_REPO_ROOT, "results", "results_logs",
                         "ffc_s5_diagnostics.log"),
    r_library = Sys.getenv("FFC_R_LIBRARY", unset = ""),
    n_bootstrap_resamples = 100L,
    n_seed_vectors = 100L,
    n_visualization_runs = 1000L,
    n_jobs = -1L,
    bootstrap_batch_size = 0L,
    fixed_seed = 8544L,
    reuse_crossed_scores = FALSE,
    help = FALSE
  )
  value_options <- c(
    "background-csv", "train-csv", "test-csv", "seed-list", "output-dir",
    "log-path", "r-library", "n-bootstrap-resamples", "n-seed-vectors",
    "n-visualization-runs", "n-jobs", "bootstrap-batch-size", "fixed-seed"
  )
  index <- 1L
  while (index <= length(arguments)) {
    option <- arguments[[index]]
    if (option == "--reuse-crossed-scores") {
      defaults$reuse_crossed_scores <- TRUE
      index <- index + 1L
      next
    }
    if (option %in% c("--help", "-h")) {
      defaults$help <- TRUE
      index <- index + 1L
      next
    }
    if (!startsWith(option, "--")) {
      stop(sprintf("unexpected positional argument: %s", option), call. = FALSE)
    }
    option_name <- substring(option, 3L)
    if (!option_name %in% value_options) {
      stop(sprintf("unknown option: %s", option), call. = FALSE)
    }
    if (index == length(arguments)) {
      stop(sprintf("option requires a value: %s", option), call. = FALSE)
    }
    key <- gsub("-", "_", option_name, fixed = TRUE)
    value <- arguments[[index + 1L]]
    if (is.integer(defaults[[key]])) {
      numeric_value <- suppressWarnings(as.numeric(value))
      if (!is.finite(numeric_value) || numeric_value != floor(numeric_value)) {
        stop(sprintf("%s requires an integer", option), call. = FALSE)
      }
      defaults[[key]] <- as.integer(numeric_value)
    } else {
      defaults[[key]] <- value
    }
    index <- index + 2L
  }
  defaults
}

.ffc_md5 <- function(path) unname(tools::md5sum(path))

.make_provenance <- function(arguments, settings, seed_plan) {
  list(
    n_bootstrap_resamples = settings$n_bootstrap_resamples,
    n_seed_vectors = settings$n_seed_vectors,
    background_md5 = .ffc_md5(arguments$background_csv),
    train_md5 = .ffc_md5(arguments$train_csv),
    test_md5 = .ffc_md5(arguments$test_csv),
    seed_list_md5 = .ffc_md5(arguments$seed_list),
    imputation_seeds = seed_plan$imputation[seq_len(settings$n_seed_vectors)],
    bootstrap_seeds = seed_plan$bootstrap,
    R_version = R.version.string,
    Amelia_version = as.character(utils::packageVersion("Amelia")),
    bootstrap_design = "stratified train/test/other row bootstrap"
  )
}

load_reused_crossed_scores <- function(output_dir, provenance) {
  score_path <- file.path(output_dir, "ffc_crossed_score_matrices.rds")
  issue_path <- file.path(output_dir, "ffc_crossed_issue_counts.rds")
  provenance_path <- file.path(output_dir, "ffc_crossed_provenance.rds")
  matrices <- readRDS(score_path)
  issues <- readRDS(issue_path)
  saved_provenance <- readRDS(provenance_path)
  if (!identical(saved_provenance, provenance)) {
    stop("saved crossed-score provenance does not match this run", call. = FALSE)
  }
  if (!is.list(matrices) || !identical(names(matrices), FFC_ESTIMANDS)) {
    stop("saved crossed matrices have unexpected estimands", call. = FALSE)
  }
  expected_dimension <- c(
    provenance$n_bootstrap_resamples, provenance$n_seed_vectors
  )
  invalid <- vapply(matrices, function(values) {
    !is.matrix(values) || !identical(dim(values), expected_dimension) ||
      any(!is.finite(values))
  }, logical(1L))
  if (any(invalid)) {
    stop(sprintf("saved crossed matrices are invalid: %s",
                 paste(names(invalid)[invalid], collapse = ", ")),
         call. = FALSE)
  }
  if (!is.numeric(issues) || !identical(names(issues), FFC_ISSUE_NAMES)) {
    stop("saved crossed issue counts are invalid", call. = FALSE)
  }
  list(matrices = matrices, issues = issues)
}

main <- function(arguments = commandArgs(trailingOnly = TRUE)) {
  arguments <- parse_ffc_args(arguments)
  if (isTRUE(arguments$help)) {
    ffc_usage()
    return(invisible(NULL))
  }
  if (nzchar(arguments$r_library)) {
    library_path <- normalizePath(arguments$r_library, mustWork = TRUE)
    .libPaths(unique(c(library_path, .libPaths())))
  }
  Sys.setenv(
    OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1",
    MKL_NUM_THREADS = "1", VECLIB_MAXIMUM_THREADS = "1"
  )
  if (!requireNamespace("Amelia", quietly = TRUE)) {
    stop(paste0(
      "The FFC estimator requires the Amelia R package. Install it with ",
      "install.packages('Amelia') or supply --r-library /path/to/library."
    ), call. = FALSE)
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite is required for the FFC diagnostics JSON", call. = FALSE)
  }
  input_paths <- c(
    arguments$background_csv, arguments$train_csv, arguments$test_csv,
    arguments$seed_list
  )
  missing_inputs <- input_paths[!file.exists(input_paths)]
  if (length(missing_inputs)) {
    stop(sprintf("missing input files: %s", paste(missing_inputs, collapse = ", ")),
         call. = FALSE)
  }
  settings <- ffc_settings(
    n_bootstrap_resamples = arguments$n_bootstrap_resamples,
    n_seed_vectors = arguments$n_seed_vectors,
    n_visualization_runs = arguments$n_visualization_runs,
    n_jobs = arguments$n_jobs,
    bootstrap_batch_size = arguments$bootstrap_batch_size,
    fixed_seed = arguments$fixed_seed
  )
  RNGkind(kind = "Mersenne-Twister", normal.kind = "Inversion",
          sample.kind = "Rejection")
  logger <- configure_run_logger("ffc_s5", arguments$log_path)
  logger$info("Starting FFC S5 analysis")
  logger$info("Background data: %s", arguments$background_csv)
  logger$info("Train outcomes: %s", arguments$train_csv)
  logger$info("Test outcomes: %s", arguments$test_csv)
  logger$info("Seed list: %s", arguments$seed_list)
  logger$info("Run-level output directory: %s", arguments$output_dir)
  logger$info(
    "Parallel workers: %d (n_jobs=%d); bootstrap batch size: %d",
    resolve_worker_count(settings$n_jobs), settings$n_jobs,
    resolve_batch_size(
      settings$n_bootstrap_resamples, settings$bootstrap_batch_size,
      settings$n_jobs
    )
  )
  started <- proc.time()[["elapsed"]]

  panel <- prepare_ffc_data(
    arguments$background_csv, arguments$train_csv, arguments$test_csv
  )
  logger$info(
    "Prepared %d FFC rows (%d train, %d test, %d other) in %.1f seconds",
    nrow(panel), sum(panel$split == "train"), sum(panel$split == "test"),
    sum(panel$split == "other"), proc.time()[["elapsed"]] - started
  )
  seeds <- load_seed_list(arguments$seed_list)
  seed_plan <- make_ffc_seed_plan(seeds, settings)
  bootstrap_indices <- stratified_bootstrap_index_block(
    panel$split, seed_plan$bootstrap
  )
  provenance <- .make_provenance(arguments, settings, seed_plan)
  logger$info(
    "Prepared %d shared stratified bootstrap datasets crossed with %d " %+%
      "imputation-seed vectors (%d cells; 18 estimands per cell)",
    settings$n_bootstrap_resamples, settings$n_seed_vectors,
    settings$n_bootstrap_resamples * settings$n_seed_vectors
  )

  if (isTRUE(arguments$reuse_crossed_scores)) {
    crossed <- load_reused_crossed_scores(arguments$output_dir, provenance)
    logger$info(
      "Reused and provenance-validated %d crossed cells for 18 estimands",
      settings$n_bootstrap_resamples * settings$n_seed_vectors
    )
  } else {
    crossed <- run_crossed_grid(
      panel, bootstrap_indices, seed_plan, settings,
      checkpoint_dir = arguments$output_dir, logger = logger
    )
  }
  observed <- run_observed_seed_sweep(panel, seed_plan, settings, logger)
  fixed_seed_result <- evaluate_ffc_seed(panel, settings$fixed_seed)
  diagnostics <- setNames(lapply(FFC_ESTIMANDS, function(estimand) {
    crossed_s5_diagnostics(crossed$matrices[[estimand]])
  }), FFC_ESTIMANDS)

  metadata <- list(
    analysis = "Fragile Families Challenge benchmarks",
    settings = settings,
    resolved_parallel_workers = resolve_worker_count(settings$n_jobs),
    input_files = list(
      background_csv = normalizePath(arguments$background_csv),
      train_csv = normalizePath(arguments$train_csv),
      test_csv = normalizePath(arguments$test_csv),
      seed_list = normalizePath(arguments$seed_list)
    ),
    input_md5 = list(
      background = provenance$background_md5,
      train = provenance$train_md5,
      test = provenance$test_md5,
      seed_list = provenance$seed_list_md5
    ),
    n_panel_rows = nrow(panel),
    split_counts = as.list(table(panel$split)),
    outcome_order = FFC_OUTCOMES,
    valid_outcome_model_pairs = split(
      .ffc_valid_pairs$account, .ffc_valid_pairs$outcome
    ),
    primary_plot_account = as.list(FFC_PRIMARY_ACCOUNT),
    versions = list(
      R = R.version.string,
      Amelia = as.character(utils::packageVersion("Amelia")),
      jsonlite = as.character(utils::packageVersion("jsonlite"))
    ),
    RNGkind = as.list(RNGkind()),
    crossed_issue_counts = as.list(crossed$issues),
    observed_issue_counts = as.list(observed$issues),
    fixed_seed_issue_counts = as.list(fixed_seed_result$issues),
    visualization_compatibility_alias = sprintf(
      "seed_analysis_10000.csv contains %d seeds",
      settings$n_visualization_runs
    )
  )
  output_objects <- write_ffc_outputs(
    arguments$output_dir, bootstrap_indices, seed_plan, crossed, observed,
    fixed_seed_result, diagnostics, metadata, provenance, settings
  )

  common_details <- list(
    within_cell_replications = paste0(
      "1 complete six-outcome pipeline; Amelia m=1 for each outcome"
    ),
    bootstrap_design = paste0(
      "crossed; the same external row resample is reused for every ",
      "imputation seed and all outcome/model estimands"
    ),
    external_dataset_bootstrap = paste0(
      "nonparametric row sampling with replacement, stratified to retain ",
      "the observed train/test/other partition sizes"
    ),
    bootstrap_PRNG = paste(
      "R", R.version.string, paste(RNGkind(), collapse = "/")
    ),
    algorithmic_PRNG = paste0(
      "one scalar R Mersenne-Twister seed drives six sequential Amelia EMB ",
      "imputations in fixed outcome order; Amelia ",
      as.character(utils::packageVersion("Amelia"))
    ),
    internal_vs_external_bootstrap = paste0(
      "Amelia's internal EMB bootstrap is algorithmic randomness and is ",
      "distinct from the shared external dataset bootstrap"
    ),
    joint_or_stagewise_variation = paste0(
      "joint scalar imputation-seed vector; deterministic lm/glm stages; ",
      "outcome order=", paste(FFC_OUTCOMES, collapse = ",")
    ),
    evaluation = paste0(
      "fixed FFC holdout; both training rows and holdout rows are externally ",
      "bootstrapped within their partitions"
    ),
    fit_failure_policy = paste0(
      "fail the run rather than silently substitute predictions; warning and ",
      "substitution counts are audited"
    ),
    parallel_backend = "parallel::mclapply fork workers; prescheduled tasks",
    parallel_workers = resolve_worker_count(settings$n_jobs),
    bootstrap_checkpoint_batch_size = resolve_batch_size(
      settings$n_bootstrap_resamples, settings$bootstrap_batch_size,
      settings$n_jobs
    ),
    crossed_amelia_warnings = crossed$issues[["amelia_warnings"]],
    crossed_ols_fit_warnings = crossed$issues[["ols_fit_warnings"]],
    crossed_logit_fit_warnings = crossed$issues[["logit_fit_warnings"]],
    crossed_fit_fallbacks = crossed$issues[["fit_fallbacks"]],
    crossed_metric_substitutions = crossed$issues[["metric_substitutions"]]
  )
  for (estimand in FFC_ESTIMANDS) {
    parts <- strsplit(estimand, "_", fixed = TRUE)[[1L]]
    outcome <- parts[[1L]]
    account <- parts[[2L]]
    metric <- parts[[3L]]
    model_description <- if (account == "ols") {
      "base R lm with lagged outcome and demographic predictors"
    } else {
      "base R glm binomial(logit) with lagged outcome and demographic predictors"
    }
    score_description <- if (metric == "R2") {
      "holdout pseudo-R2 relative to the observed training-outcome mean"
    } else {
      paste0("coefficient on lagged ", outcome, " predictor")
    }
    details <- c(common_details, list(
      outcome = outcome,
      model = model_description,
      score = score_description
    ))
    logger$info("\n%s", format_s5_report(
      diagnostics[[estimand]], estimand, details
    ))
  }
  for (estimand in FFC_ESTIMANDS) {
    logger$info("\n%s", format_observed_seed_report(
      observed_seed_summary(observed$scores[, estimand]), estimand
    ))
  }
  logger$info(
    "Crossed issue counts: %s",
    paste(sprintf("%s=%d", names(crossed$issues), crossed$issues),
          collapse = ", ")
  )
  logger$info(
    "Observed issue counts: %s",
    paste(sprintf("%s=%d", names(observed$issues), observed$issues),
          collapse = ", ")
  )
  logger$info(
    "Fixed-seed issue counts: %s",
    paste(sprintf("%s=%d", names(fixed_seed_result$issues),
                  fixed_seed_result$issues), collapse = ", ")
  )
  warning_positions <- c("amelia_warnings", "ols_fit_warnings",
                         "logit_fit_warnings")
  if (sum(crossed$issues[warning_positions]) > 0L) {
    logger$warning(
      "The crossed grid emitted model/imputation warnings; counts are reported above"
    )
  }
  if (crossed$issues[["fit_fallbacks"]] > 0L ||
      crossed$issues[["metric_substitutions"]] > 0L) {
    logger$warning(
      "The crossed grid used a fallback/substitution; interpret diagnostics cautiously"
    )
  }
  logger$info(
    "Visualization output: %d rows (%d seeds; compatibility alias also written)",
    nrow(output_objects$visualization_frame), settings$n_visualization_runs
  )
  validate_s5_log(arguments$log_path, FFC_ESTIMANDS)
  logger$info("FFC analysis complete in %.1f seconds",
              proc.time()[["elapsed"]] - started)
  invisible(list(
    diagnostics = diagnostics,
    crossed = crossed,
    observed = observed,
    fixed_seed = fixed_seed_result,
    outputs = output_objects
  ))
}

if (sys.nframe() == 0L) {
  main()
}
