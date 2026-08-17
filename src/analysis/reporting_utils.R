# Reusable reporting helpers for seed-sensitivity experiments.
#
# Crossed score matrices always use bootstrap datasets on rows (B) and
# algorithmic seed vectors on columns (m).  The formulas and finite-sample
# boundary policy mirror reporting_utils.py.

.assert_scalar_integer <- function(value, name, minimum = 0L) {
  if (length(value) != 1L || !is.numeric(value) || !is.finite(value) ||
      value != floor(value) || value < minimum || value > .Machine$integer.max) {
    stop(sprintf("%s must be one integer in [%d, %d]", name, minimum,
                 .Machine$integer.max), call. = FALSE)
  }
  as.integer(value)
}

.validate_seeds <- function(seeds, name = "seeds") {
  if (!is.numeric(seeds) || length(seeds) == 0L || any(!is.finite(seeds)) ||
      any(seeds != floor(seeds)) || any(seeds < 0) ||
      any(seeds > .Machine$integer.max)) {
    stop(sprintf("%s must contain R-compatible integer seeds in [0, %d]",
                 name, .Machine$integer.max), call. = FALSE)
  }
  as.integer(seeds)
}

load_seed_list <- function(path) {
  lines <- trimws(readLines(path, warn = FALSE, encoding = "UTF-8"))
  lines <- lines[nzchar(lines) & !startsWith(lines, "#")]
  if (length(lines) == 0L) {
    stop(sprintf("seed list is empty: %s", path), call. = FALSE)
  }
  values <- suppressWarnings(as.numeric(lines))
  if (anyNA(values)) {
    bad <- which(is.na(values))[1L]
    stop(sprintf("invalid seed on retained line %d of %s", bad, path),
         call. = FALSE)
  }
  .validate_seeds(values, "seed list")
}

seed_component_blocks <- function(seeds, n_vectors, component_names,
                                  offset = 0L) {
  seeds <- .validate_seeds(seeds)
  n_vectors <- .assert_scalar_integer(n_vectors, "n_vectors", 1L)
  offset <- .assert_scalar_integer(offset, "offset", 0L)
  if (!is.character(component_names) || length(component_names) == 0L ||
      any(!nzchar(component_names)) || anyDuplicated(component_names)) {
    stop("component_names must be non-empty and unique", call. = FALSE)
  }
  required <- n_vectors * length(component_names)
  stop_position <- offset + required
  if (stop_position > length(seeds)) {
    stop(sprintf(
      "seed list has %d values but zero-based positions %d:%d are required",
      length(seeds), offset, stop_position
    ), call. = FALSE)
  }
  result <- lapply(seq_along(component_names), function(index) {
    start <- offset + (index - 1L) * n_vectors + 1L
    seeds[start:(start + n_vectors - 1L)]
  })
  names(result) <- component_names
  result
}

.with_preserved_rng <- function(code) {
  old_kind <- RNGkind()
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) {
    old_seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  }
  on.exit({
    do.call(RNGkind, as.list(old_kind))
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  RNGkind(kind = "Mersenne-Twister", normal.kind = "Inversion",
          sample.kind = "Rejection")
  force(code)
}

bootstrap_index_block <- function(n_observations, bootstrap_seeds,
                                  sample_size = NULL) {
  n_observations <- .assert_scalar_integer(
    n_observations, "n_observations", 1L
  )
  bootstrap_seeds <- .validate_seeds(bootstrap_seeds, "bootstrap_seeds")
  if (is.null(sample_size)) {
    sample_size <- n_observations
  }
  sample_size <- .assert_scalar_integer(sample_size, "sample_size", 1L)
  .with_preserved_rng({
    indices <- matrix(NA_integer_, nrow = length(bootstrap_seeds),
                      ncol = sample_size)
    for (row in seq_along(bootstrap_seeds)) {
      set.seed(bootstrap_seeds[[row]])
      indices[row, ] <- sample.int(n_observations, sample_size, replace = TRUE)
    }
    indices
  })
}

stratified_bootstrap_index_block <- function(strata, bootstrap_seeds) {
  if (length(strata) == 0L) {
    stop("strata must be non-empty", call. = FALSE)
  }
  bootstrap_seeds <- .validate_seeds(bootstrap_seeds, "bootstrap_seeds")
  labels <- as.character(strata)
  labels[is.na(labels)] <- "<NA>"
  levels_in_order <- unique(labels)
  positions <- lapply(levels_in_order, function(level) which(labels == level))
  .with_preserved_rng({
    indices <- matrix(NA_integer_, nrow = length(bootstrap_seeds),
                      ncol = length(labels))
    for (row in seq_along(bootstrap_seeds)) {
      set.seed(bootstrap_seeds[[row]])
      for (group_positions in positions) {
        # Index through sample.int() to avoid base::sample()'s length-one
        # numeric special case (sample(5, 1) draws from 1:5).
        draw <- sample.int(
          length(group_positions), length(group_positions), replace = TRUE
        )
        indices[row, group_positions] <- group_positions[draw]
      }
    }
    indices
  })
}

full_factorial_seed_grid <- function(first_component, second_component,
                                     n_runs) {
  n_runs <- .assert_scalar_integer(n_runs, "n_runs", 1L)
  first_component <- .validate_seeds(first_component, "first_component")
  second_component <- .validate_seeds(second_component, "second_component")
  n_second <- floor(sqrt(n_runs))
  while (n_runs %% n_second != 0L) {
    n_second <- n_second - 1L
  }
  n_first <- n_runs %/% n_second
  if (length(first_component) < n_first ||
      length(second_component) < n_second) {
    stop(sprintf(
      "seed components are too short: need %d and %d, got %d and %d",
      n_first, n_second, length(first_component), length(second_component)
    ), call. = FALSE)
  }
  list(
    first = rep(first_component[seq_len(n_first)], each = n_second),
    second = rep(second_component[seq_len(n_second)], times = n_first),
    n_first = n_first,
    n_second = n_second
  )
}

resolve_worker_count <- function(n_jobs, n_tasks = NULL) {
  if (length(n_jobs) != 1L || !is.numeric(n_jobs) || !is.finite(n_jobs) ||
      n_jobs != floor(n_jobs) || n_jobs == 0L || n_jobs < -1L) {
    stop("n_jobs must be -1 (all available CPUs) or a positive integer",
         call. = FALSE)
  }
  workers <- if (n_jobs == -1L) {
    detected <- parallel::detectCores(logical = TRUE)
    if (is.na(detected)) 1L else as.integer(detected)
  } else {
    as.integer(n_jobs)
  }
  if (!is.null(n_tasks)) {
    n_tasks <- .assert_scalar_integer(n_tasks, "n_tasks", 1L)
    workers <- min(workers, n_tasks)
  }
  max(1L, workers)
}

resolve_batch_size <- function(n_items, batch_size, n_jobs) {
  n_items <- .assert_scalar_integer(n_items, "n_items", 1L)
  batch_size <- .assert_scalar_integer(batch_size, "batch_size", 0L)
  if (batch_size == 0L) {
    return(resolve_worker_count(n_jobs, n_tasks = n_items))
  }
  min(n_items, batch_size)
}

crossed_s5_diagnostics <- function(score_matrix) {
  if (!is.matrix(score_matrix) || !is.numeric(score_matrix) ||
      length(dim(score_matrix)) != 2L || min(dim(score_matrix)) < 2L) {
    stop(paste(
      "score_matrix must be a numeric matrix with at least two rows",
      "and two columns"
    ), call. = FALSE)
  }
  scores <- matrix(as.numeric(score_matrix), nrow = nrow(score_matrix),
                   ncol = ncol(score_matrix))
  if (any(!is.finite(scores))) {
    stop("score_matrix must be complete and finite", call. = FALSE)
  }
  B <- nrow(scores)
  m <- ncol(scores)
  grand_mean <- mean(scores)
  bootstrap_means <- rowMeans(scores)
  seed_means <- colMeans(scores)

  data_variance <- mean(apply(scores, 2L, var))
  seed_ms <- B * var(seed_means)
  data_ms <- m * var(bootstrap_means)
  residual <- scores - outer(bootstrap_means, rep(1, m)) -
    outer(rep(1, B), seed_means) + grand_mean
  interaction_ms <- sum(residual^2) / ((B - 1L) * (m - 1L))

  between_seed_variance_raw <- (seed_ms - interaction_ms) / B
  between_seed_variance <- max(0, between_seed_variance_raw)
  boundary_hit <- between_seed_variance_raw < 0
  data_main_effect_variance <- (data_ms - interaction_ms) / m
  identity_difference <- abs(
    data_variance - (data_main_effect_variance + interaction_ms)
  )
  identity_tolerance <- 1e-12 + 1e-10 * abs(
    data_main_effect_variance + interaction_ms
  )
  if (identity_difference > identity_tolerance) {
    stop("crossed-grid variance identity failed", call. = FALSE)
  }

  data_sd <- sqrt(data_variance)
  seed_sd <- sqrt(between_seed_variance)
  relative <- if (data_sd == 0) {
    if (seed_sd == 0) NA_real_ else Inf
  } else {
    seed_sd / data_sd
  }
  denominator <- between_seed_variance + data_variance
  if (denominator == 0) {
    first_order_share <- NA_real_
    total_order_share <- NA_real_
  } else {
    first_order_share <- between_seed_variance / denominator
    total_order_share <- if (data_main_effect_variance < 0) {
      NA_real_
    } else {
      (between_seed_variance + interaction_ms) / denominator
    }
  }

  structure(list(
    n_bootstrap_resamples = B,
    n_seed_vectors = m,
    seed_averaged_estimate = grand_mean,
    data_variance = data_variance,
    data_uncertainty_sd = data_sd,
    between_seed_variance_raw = between_seed_variance_raw,
    between_seed_variance = between_seed_variance,
    variance_component_boundary_hit = boundary_hit,
    between_seed_variability_sd = seed_sd,
    relative_importance = relative,
    algorithmic_variance_share = first_order_share,
    total_order_algorithmic_variance_share = total_order_share,
    data_main_effect_variance = data_main_effect_variance,
    data_seed_interaction_variance = interaction_ms,
    seed_mean_square = seed_ms,
    data_mean_square = data_ms,
    interaction_mean_square = interaction_ms
  ), class = c("s5_diagnostics", "list"))
}

observed_seed_summary <- function(scores) {
  if (!is.numeric(scores) || is.matrix(scores) || length(scores) < 2L ||
      any(!is.finite(scores))) {
    stop("scores must be a finite numeric vector of length >= 2", call. = FALSE)
  }
  list(
    n_seed_vectors = length(scores),
    observed_data_seed_average = mean(scores),
    observed_between_seed_sd = sd(scores),
    minimum = min(scores),
    maximum = max(scores)
  )
}

.display_number <- function(value) {
  if (length(value) == 0L || is.null(value) ||
      (length(value) == 1L && is.na(value))) {
    return("not_estimable")
  }
  if (is.logical(value) && length(value) == 1L) {
    return(tolower(as.character(value)))
  }
  if (is.numeric(value) && length(value) == 1L) {
    return(sprintf("%.12g", value))
  }
  paste(as.character(value), collapse = "; ")
}

format_s5_report <- function(diagnostics, estimand,
                             computational_details = list()) {
  d <- diagnostics
  lines <- c(
    sprintf("S5 recommended reporting and diagnostics: %s", estimand),
    "1. Seed-averaged estimate",
    sprintf("   theta_bar=%s", .display_number(d$seed_averaged_estimate)),
    "2. Data uncertainty",
    sprintf("   SD_theta_D=%s", .display_number(d$data_uncertainty_sd)),
    sprintf("   sigma_D_squared=%s", .display_number(d$data_variance)),
    "3. Bias-corrected between-seed variability",
    sprintf("   SD_theta_S=%s", .display_number(d$between_seed_variability_sd)),
    sprintf("   sigma_S_squared_adj_nonnegative=%s",
            .display_number(d$between_seed_variance)),
    sprintf("   sigma_S_squared_adj_raw=%s",
            .display_number(d$between_seed_variance_raw)),
    sprintf("   variance_component_boundary_hit=%s",
            .display_number(d$variance_component_boundary_hit)),
    "4. Relative importance of algorithmic randomness",
    sprintf("   r=%s", .display_number(d$relative_importance)),
    "5. Algorithmic variance share",
    sprintf("   rho_S=%s", .display_number(d$algorithmic_variance_share)),
    "   Total-order algorithmic variance share",
    sprintf("   rho_T_S=%s",
            .display_number(d$total_order_algorithmic_variance_share)),
    "6. Computational details",
    sprintf("   m_seed_vectors=%d", d$n_seed_vectors),
    sprintf("   B_bootstrap_resamples=%d", d$n_bootstrap_resamples)
  )
  if (length(computational_details)) {
    if (is.null(names(computational_details)) ||
        any(!nzchar(names(computational_details)))) {
      stop("computational_details must be a named list", call. = FALSE)
    }
    detail_lines <- vapply(seq_along(computational_details), function(index) {
      sprintf("   %s=%s", names(computational_details)[[index]],
              .display_number(computational_details[[index]]))
    }, character(1L))
    lines <- c(lines, detail_lines)
  }
  lines <- c(
    lines,
    "Supporting crossed-grid diagnostics",
    sprintf("   V_D_hat=%s", .display_number(d$data_main_effect_variance)),
    sprintf("   V_DS_hat=%s",
            .display_number(d$data_seed_interaction_variance)),
    sprintf("   MS_S=%s", .display_number(d$seed_mean_square)),
    sprintf("   MS_D=%s", .display_number(d$data_mean_square)),
    sprintf("   MS_int=%s", .display_number(d$interaction_mean_square))
  )
  if (isTRUE(d$variance_component_boundary_hit)) {
    lines <- c(lines, paste0(
      "   warning=The unconstrained equation-(27) seed-variance estimate ",
      "is slightly negative. S5 point estimates use the explicitly reported ",
      "nonnegative boundary value zero; the raw estimate is retained above. ",
      "A boundary point estimate does not prove that the true seed main-effect ",
      "variance is exactly zero."
    ))
  }
  if (d$data_main_effect_variance < 0) {
    lines <- c(lines, paste0(
      "   warning=The finite-sample data main-effect estimate is negative; ",
      "the total-order share would lie outside [0, 1], so rho_T_S is not ",
      "reported and was not silently clipped."
    ))
  }
  paste(lines, collapse = "\n")
}

format_observed_seed_report <- function(summary, estimand) {
  paste(c(
    sprintf("Observed-data seed sweep (no bootstrap): %s", estimand),
    sprintf("n_seed_vectors=%d", summary$n_seed_vectors),
    sprintf("observed_data_seed_average=%s",
            .display_number(summary$observed_data_seed_average)),
    sprintf("observed_between_seed_sd_conditional_on_D_obs=%s",
            .display_number(summary$observed_between_seed_sd)),
    sprintf("minimum=%s", .display_number(summary$minimum)),
    sprintf("maximum=%s", .display_number(summary$maximum)),
    paste0("interpretation=Conditional variability on the single observed ",
           "dataset; it estimates the total seed effect V_S + V_DS."),
    paste0("not_reported=Data uncertainty, r, and rho_S are not identified ",
           "without resampling and are not computed from this sweep.")
  ), collapse = "\n")
}

configure_run_logger <- function(name, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  writeLines(character(), path, useBytes = TRUE)
  emit <- function(level, format, ...) {
    arguments <- list(...)
    text <- if (length(arguments)) {
      do.call(sprintf, c(list(format), arguments))
    } else {
      as.character(format)
    }
    entry <- sprintf("%s | %s | %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
                     level, text)
    cat(entry, "\n", file = stderr(), sep = "")
    cat(entry, "\n", file = path, append = TRUE, sep = "")
    invisible(entry)
  }
  list(
    name = name,
    path = path,
    info = function(format, ...) emit("INFO", format, ...),
    warning = function(format, ...) emit("WARNING", format, ...),
    error = function(format, ...) emit("ERROR", format, ...)
  )
}

validate_s5_log <- function(path, estimands) {
  if (!is.character(estimands) || length(estimands) == 0L ||
      any(!nzchar(estimands))) {
    stop("estimands must be a non-empty character vector", call. = FALSE)
  }
  text <- paste(readLines(path, warn = FALSE, encoding = "UTF-8"),
                collapse = "\n")
  markers <- paste0("S5 recommended reporting and diagnostics: ", estimands)
  starts <- vapply(markers, function(marker) {
    match <- regexpr(marker, text, fixed = TRUE)[[1L]]
    if (match < 0L) {
      stop(sprintf("S5 log is missing estimand block: %s", marker),
           call. = FALSE)
    }
    match
  }, integer(1L))
  headings <- c(
    "1. Seed-averaged estimate",
    "2. Data uncertainty",
    "3. Bias-corrected between-seed variability",
    "4. Relative importance of algorithmic randomness",
    "5. Algorithmic variance share",
    "6. Computational details"
  )
  for (index in seq_along(markers)) {
    later <- starts[starts > starts[[index]]]
    stop_position <- if (length(later)) min(later) - 1L else nchar(text)
    block <- substr(text, starts[[index]], stop_position)
    missing <- headings[!vapply(
      headings, grepl, logical(1L), x = block, fixed = TRUE
    )]
    if (length(missing)) {
      stop(sprintf("S5 log block for %s is incomplete; missing: %s",
                   markers[[index]], paste(missing, collapse = ", ")),
           call. = FALSE)
    }
  }
  invisible(TRUE)
}

json_safe <- function(value) {
  if (is.data.frame(value)) {
    result <- value
    for (name in names(result)) {
      result[[name]] <- json_safe(result[[name]])
    }
    return(result)
  }
  if (is.list(value)) {
    return(lapply(value, json_safe))
  }
  if (is.numeric(value)) {
    value[!is.finite(value)] <- NA_real_
  }
  value
}

write_strict_json <- function(value, path) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite is required to write strict JSON", call. = FALSE)
  }
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  text <- jsonlite::toJSON(
    json_safe(value), pretty = TRUE, auto_unbox = TRUE, na = "null",
    null = "null", digits = NA
  )
  writeLines(c(text, ""), path, useBytes = TRUE)
  invisible(path)
}

atomic_save_rds <- function(value, path, compress = FALSE) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- paste0(path, ".tmp")
  saveRDS(value, temporary, compress = compress)
  if (!file.rename(temporary, path)) {
    unlink(temporary)
    stop(sprintf("could not atomically replace %s", path), call. = FALSE)
  }
  invisible(path)
}
