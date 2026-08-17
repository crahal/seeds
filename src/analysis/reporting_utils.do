version 17.0

/*
Reusable S5 reporting helpers for Stata builders.

The crossed score table is long but represents a balanced B x m matrix:
bootstrap() identifies rows D_b* and seed() identifies algorithmic columns.
The formulas mirror reporting_utils.py and reporting_utils.R.  In particular,
the unconstrained equation-(27) seed component is retained for audit, while
the primary standard deviation and variance shares use its explicitly
reported nonnegative boundary value max(V_S, 0).  A negative V_D estimate
makes the total-order share not estimable rather than silently clipping it.
*/

capture program drop s5__display_number
program define s5__display_number, rclass
    version 17.0
    syntax, VALUE(real)
    if missing(`value') {
        if `value' == .i {
            return local text "infinite"
        }
        else {
            return local text "not_estimable"
        }
    }
    else {
        local rendered : display %21.12g `value'
        local cleaned = strtrim(`"`rendered'"')
        return local text `"`cleaned'"'
    }
end


capture program drop s5_log_line
program define s5_log_line
    version 17.0
    syntax using/, TEXT(string asis) [REPLACE]
    tempname log_handle
    if `"`replace'"' != "" {
        file open `log_handle' using `"`using'"', write replace text
    }
    else {
        file open `log_handle' using `"`using'"', write append text
    }
    file write `log_handle' `"`text'"' _n
    file close `log_handle'
end


capture program drop s5_crossed_diagnostics
program define s5_crossed_diagnostics, rclass
    version 17.0
    syntax [if] [in], SCORE(varname numeric) ///
        BOOTstrap(varname numeric) SEED(varname numeric)
    marksample touse, novarlist
    preserve
    quietly keep if `touse'

    quietly count
    local n_cells = r(N)
    quietly count if missing(`score') | missing(`bootstrap') | missing(`seed')
    if r(N) {
        restore
        display as error "The crossed score grid must be complete and finite"
        exit 459
    }
    capture isid `bootstrap' `seed'
    if _rc {
        restore
        display as error "bootstrap() and seed() must uniquely identify cells"
        exit 459
    }

    tempvar tag_bootstrap tag_seed bootstrap_mean seed_mean seed_sd ///
        seed_variance residual residual_squared residual_ss
    quietly egen byte `tag_bootstrap' = tag(`bootstrap')
    quietly egen byte `tag_seed' = tag(`seed')
    quietly count if `tag_bootstrap'
    local B = r(N)
    quietly count if `tag_seed'
    local m = r(N)
    if `B' < 2 | `m' < 2 | `n_cells' != `B' * `m' {
        restore
        display as error "The crossed grid must be balanced with B>=2 and m>=2"
        exit 459
    }
    quietly bysort `bootstrap': assert _N == `m'
    quietly bysort `seed': assert _N == `B'

    quietly summarize `score', meanonly
    local grand_mean = r(mean)
    quietly bysort `bootstrap': egen double `bootstrap_mean' = mean(`score')
    quietly bysort `seed': egen double `seed_mean' = mean(`score')

    /* Equation (25): average fixed-seed sample variances over D_b*. */
    quietly bysort `seed': egen double `seed_sd' = sd(`score')
    quietly generate double `seed_variance' = `seed_sd'^2 if `tag_seed'
    quietly summarize `seed_variance', meanonly
    local data_variance = r(mean)

    /* Equations (26)-(27): balanced two-way random-effects mean squares. */
    quietly summarize `seed_mean' if `tag_seed'
    local seed_ms = `B' * r(Var)
    quietly summarize `bootstrap_mean' if `tag_bootstrap'
    local data_ms = `m' * r(Var)
    quietly generate double `residual' = `score' - `bootstrap_mean' - ///
        `seed_mean' + `grand_mean'
    quietly generate double `residual_squared' = `residual'^2
    quietly egen double `residual_ss' = total(`residual_squared')
    quietly summarize `residual_ss', meanonly
    local interaction_ms = r(min) / ((`B' - 1) * (`m' - 1))

    local seed_variance_raw = (`seed_ms' - `interaction_ms') / `B'
    local seed_variance_report = max(0, `seed_variance_raw')
    local boundary_hit = (`seed_variance_raw' < 0)
    local data_main_variance = (`data_ms' - `interaction_ms') / `m'
    local identity_rhs = `data_main_variance' + `interaction_ms'
    local identity_tolerance = max(1e-12, ///
        1e-10 * max(abs(`data_variance'), abs(`identity_rhs')))
    if abs(`data_variance' - `identity_rhs') > `identity_tolerance' {
        restore
        display as error "Crossed-grid variance identity failed"
        exit 459
    }

    local data_sd = sqrt(max(0, `data_variance'))
    local seed_sd_report = sqrt(`seed_variance_report')
    local relative_importance = .
    if `data_sd' > 0 {
        local relative_importance = `seed_sd_report' / `data_sd'
    }
    else if `seed_sd_report' > 0 {
        /* Stata has no IEEE infinity; .i is formatted explicitly as infinite. */
        local relative_importance = .i
    }

    local denominator = `seed_variance_report' + `data_variance'
    local first_order_share = .
    local total_order_share = .
    if `denominator' > 0 {
        local first_order_share = `seed_variance_report' / `denominator'
        if `data_main_variance' >= 0 {
            local total_order_share = ///
                (`seed_variance_report' + `interaction_ms') / `denominator'
        }
    }

    restore
    return scalar n_bootstrap_resamples = `B'
    return scalar n_seed_vectors = `m'
    return scalar seed_averaged_estimate = `grand_mean'
    return scalar data_variance = `data_variance'
    return scalar data_uncertainty_sd = `data_sd'
    return scalar between_seed_variance_raw = `seed_variance_raw'
    return scalar between_seed_variance = `seed_variance_report'
    return scalar variance_component_boundary_hit = `boundary_hit'
    return scalar between_seed_variability_sd = `seed_sd_report'
    return scalar relative_importance = `relative_importance'
    return scalar algorithmic_variance_share = `first_order_share'
    /* Full Python/R name exceeds Stata's 32-character identifier limit. */
    return scalar total_order_algorithmic_share = `total_order_share'
    return scalar data_main_effect_variance = `data_main_variance'
    return scalar data_seed_interaction_variance = `interaction_ms'
    return scalar seed_mean_square = `seed_ms'
    return scalar data_mean_square = `data_ms'
    return scalar interaction_mean_square = `interaction_ms'
end


capture program drop s5_observed_summary
program define s5_observed_summary, rclass
    version 17.0
    syntax [if] [in], SCORE(varname numeric)
    marksample touse, novarlist
    quietly count if `touse'
    if r(N) < 2 {
        display as error "Observed-data scores must contain at least two values"
        exit 459
    }
    quietly count if `touse' & missing(`score')
    if r(N) {
        display as error "Observed-data scores must be finite and complete"
        exit 459
    }
    quietly summarize `score' if `touse'
    return scalar n_seed_vectors = r(N)
    return scalar observed_data_seed_average = r(mean)
    return scalar observed_between_seed_sd = r(sd)
    return scalar minimum = r(min)
    return scalar maximum = r(max)
end


capture program drop s5_write_report
program define s5_write_report, rclass
    version 17.0
    syntax using/, SCORE(varname numeric) BOOTstrap(varname numeric) ///
        SEED(varname numeric) ESTimand(string asis) [REPLACE ///
        WITHINrep(string asis) PRNG(string asis) ///
        VARIATION(string asis) DETAILS(string asis)]

    s5_crossed_diagnostics, score(`score') bootstrap(`bootstrap') seed(`seed')
    tempname B m theta data_variance data_sd seed_raw seed_report boundary ///
        seed_sd relative rho_s rho_t v_d v_ds ms_s ms_d ms_int
    scalar `B' = r(n_bootstrap_resamples)
    scalar `m' = r(n_seed_vectors)
    scalar `theta' = r(seed_averaged_estimate)
    scalar `data_variance' = r(data_variance)
    scalar `data_sd' = r(data_uncertainty_sd)
    scalar `seed_raw' = r(between_seed_variance_raw)
    scalar `seed_report' = r(between_seed_variance)
    scalar `boundary' = r(variance_component_boundary_hit)
    scalar `seed_sd' = r(between_seed_variability_sd)
    scalar `relative' = r(relative_importance)
    scalar `rho_s' = r(algorithmic_variance_share)
    scalar `rho_t' = r(total_order_algorithmic_share)
    scalar `v_d' = r(data_main_effect_variance)
    scalar `v_ds' = r(data_seed_interaction_variance)
    scalar `ms_s' = r(seed_mean_square)
    scalar `ms_d' = r(data_mean_square)
    scalar `ms_int' = r(interaction_mean_square)

    if `"`withinrep'"' == "" local withinrep "1"
    if `"`prng'"' == "" local prng "not_recorded"
    if `"`variation'"' == "" local variation "not_recorded"

    if `"`replace'"' != "" {
        s5_log_line using `"`using'"', ///
            text("S5 recommended reporting and diagnostics: `estimand'") replace
    }
    else {
        s5_log_line using `"`using'"', ///
            text("S5 recommended reporting and diagnostics: `estimand'")
    }

    s5__display_number, value(`=scalar(`theta')')
    local theta_text `"`r(text)'"'
    s5_log_line using `"`using'"', text("1. Seed-averaged estimate")
    s5_log_line using `"`using'"', text("   theta_bar=`theta_text'")

    s5__display_number, value(`=scalar(`data_sd')')
    local data_sd_text `"`r(text)'"'
    s5__display_number, value(`=scalar(`data_variance')')
    local data_variance_text `"`r(text)'"'
    s5_log_line using `"`using'"', text("2. Data uncertainty")
    s5_log_line using `"`using'"', text("   SD_theta_D=`data_sd_text'")
    s5_log_line using `"`using'"', ///
        text("   sigma_D_squared=`data_variance_text'")

    s5__display_number, value(`=scalar(`seed_sd')')
    local seed_sd_text `"`r(text)'"'
    s5__display_number, value(`=scalar(`seed_report')')
    local seed_report_text `"`r(text)'"'
    s5__display_number, value(`=scalar(`seed_raw')')
    local seed_raw_text `"`r(text)'"'
    local boundary_text = cond(scalar(`boundary') != 0, "true", "false")
    s5_log_line using `"`using'"', ///
        text("3. Bias-corrected between-seed variability")
    s5_log_line using `"`using'"', text("   SD_theta_S=`seed_sd_text'")
    s5_log_line using `"`using'"', ///
        text("   sigma_S_squared_adj_nonnegative=`seed_report_text'")
    s5_log_line using `"`using'"', ///
        text("   sigma_S_squared_adj_raw=`seed_raw_text'")
    s5_log_line using `"`using'"', ///
        text("   variance_component_boundary_hit=`boundary_text'")

    s5__display_number, value(`=scalar(`relative')')
    local relative_text `"`r(text)'"'
    s5_log_line using `"`using'"', ///
        text("4. Relative importance of algorithmic randomness")
    s5_log_line using `"`using'"', text("   r=`relative_text'")

    s5__display_number, value(`=scalar(`rho_s')')
    local rho_s_text `"`r(text)'"'
    s5__display_number, value(`=scalar(`rho_t')')
    local rho_t_text `"`r(text)'"'
    s5_log_line using `"`using'"', text("5. Algorithmic variance share")
    s5_log_line using `"`using'"', text("   rho_S=`rho_s_text'")
    s5_log_line using `"`using'"', ///
        text("   Total-order algorithmic variance share")
    s5_log_line using `"`using'"', text("   rho_T_S=`rho_t_text'")

    s5_log_line using `"`using'"', text("6. Computational details")
    s5_log_line using `"`using'"', ///
        text("   m_seed_vectors=`=scalar(`m')'")
    s5_log_line using `"`using'"', ///
        text("   B_bootstrap_resamples=`=scalar(`B')'")
    s5_log_line using `"`using'"', ///
        text("   within_cell_replication=`withinrep'")
    s5_log_line using `"`using'"', text("   prng=`prng'")
    s5_log_line using `"`using'"', text("   variation=`variation'")
    if `"`details'"' != "" {
        s5_log_line using `"`using'"', text("   details=`details'")
    }

    s5__display_number, value(`=scalar(`v_d')')
    local v_d_text `"`r(text)'"'
    s5__display_number, value(`=scalar(`v_ds')')
    local v_ds_text `"`r(text)'"'
    s5__display_number, value(`=scalar(`ms_s')')
    local ms_s_text `"`r(text)'"'
    s5__display_number, value(`=scalar(`ms_d')')
    local ms_d_text `"`r(text)'"'
    s5__display_number, value(`=scalar(`ms_int')')
    local ms_int_text `"`r(text)'"'
    s5_log_line using `"`using'"', text("Supporting crossed-grid diagnostics")
    s5_log_line using `"`using'"', text("   V_D_hat=`v_d_text'")
    s5_log_line using `"`using'"', text("   V_DS_hat=`v_ds_text'")
    s5_log_line using `"`using'"', text("   MS_S=`ms_s_text'")
    s5_log_line using `"`using'"', text("   MS_D=`ms_d_text'")
    s5_log_line using `"`using'"', text("   MS_int=`ms_int_text'")
    if scalar(`boundary') != 0 {
        s5_log_line using `"`using'"', ///
            text("   warning=The unconstrained equation-(27) seed variance is negative. Primary S5 quantities use the explicitly reported nonnegative boundary value zero; the raw estimate is retained and zero does not prove the true seed variance is zero.")
    }
    if scalar(`v_d') < 0 {
        s5_log_line using `"`using'"', ///
            text("   warning=The finite-sample data main-effect estimate is negative; rho_T_S is not reported and was not silently clipped.")
    }

    return scalar n_bootstrap_resamples = scalar(`B')
    return scalar n_seed_vectors = scalar(`m')
    return scalar seed_averaged_estimate = scalar(`theta')
    return scalar data_variance = scalar(`data_variance')
    return scalar data_uncertainty_sd = scalar(`data_sd')
    return scalar between_seed_variance_raw = scalar(`seed_raw')
    return scalar between_seed_variance = scalar(`seed_report')
    return scalar variance_component_boundary_hit = scalar(`boundary')
    return scalar between_seed_variability_sd = scalar(`seed_sd')
    return scalar relative_importance = scalar(`relative')
    return scalar algorithmic_variance_share = scalar(`rho_s')
    return scalar total_order_algorithmic_share = scalar(`rho_t')
    return scalar data_main_effect_variance = scalar(`v_d')
    return scalar data_seed_interaction_variance = scalar(`v_ds')
    return scalar seed_mean_square = scalar(`ms_s')
    return scalar data_mean_square = scalar(`ms_d')
    return scalar interaction_mean_square = scalar(`ms_int')
end


capture program drop s5_validate_log
program define s5_validate_log, rclass
    version 17.0
    syntax using/, ESTimands(string asis)
    capture confirm file `"`using'"'
    if _rc {
        display as error "S5 log does not exist: `using'"
        exit 601
    }
    tempname log_handle
    file open `log_handle' using `"`using'"', read text
    file read `log_handle' line
    local contents ""
    while r(eof) == 0 {
        local contents `"`contents' `line'"'
        file read `log_handle' line
    }
    file close `log_handle'

    local remaining `"`estimands'"'
    while `"`remaining'"' != "" {
        local separator = strpos(`"`remaining'"', "|")
        if `separator' == 0 {
            local current = strtrim(`"`remaining'"')
            local remaining ""
        }
        else {
            local current = strtrim(substr(`"`remaining'"', 1, `separator' - 1))
            local remaining = substr(`"`remaining'"', `separator' + 1, .)
        }
        local marker "S5 recommended reporting and diagnostics: `current'"
        local block_start = strpos(`"`contents'"', `"`marker'"')
        if `"`current'"' != "" & `block_start' == 0 {
            display as error "S5 log is missing estimand block: `current'"
            exit 459
        }
        if `"`current'"' != "" {
            local after_marker = substr(`"`contents'"', ///
                `block_start' + strlen(`"`marker'"'), .)
            local next_block = strpos(`"`after_marker'"', ///
                "S5 recommended reporting and diagnostics:")
            if `next_block' == 0 {
                local block = substr(`"`contents'"', `block_start', .)
            }
            else {
                local block = substr(`"`contents'"', `block_start', ///
                    strlen(`"`marker'"') + `next_block' - 1)
            }
            foreach heading in ///
                "1. Seed-averaged estimate" ///
                "2. Data uncertainty" ///
                "3. Bias-corrected between-seed variability" ///
                "4. Relative importance of algorithmic randomness" ///
                "5. Algorithmic variance share" ///
                "6. Computational details" {
                if strpos(`"`block'"', `"`heading'"') == 0 {
                    display as error "S5 log block for `current' is missing: `heading'"
                    exit 459
                }
            }
        }
    }
    return scalar valid = 1
end
