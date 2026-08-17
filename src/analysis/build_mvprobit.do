version 17.0

/*
MVProbit seed- and sampling-uncertainty analysis.

Two experiments are intentionally separate:
  1. S5: B=100 bootstrapped datasets crossed with m=100 mvprobit seeds at
     the manuscript's two reported configurations, draws=2 and draws=150.
     Each sampled dataset is stored once and reused across seeds/configurations.
  2. Plot input: 1,000 seeds on the observed data at draws 2,4,...,150.
     This sweep is never used for data uncertainty, r, rho_S, or rho_T_S.

Run from the repository root:
    stata -b do src/analysis/build_mvprobit.do
Resume complete bootstrap/seed checkpoints with:
    stata -b do src/analysis/build_mvprobit.do "/path/to/seeds" resume
*/

clear all
set more off
set varabbrev off
set linesize 255
set rng mt64

args requested_root run_mode
if `"`run_mode'"' == "" local run_mode "fresh"
local run_mode = lower(strtrim(`"`run_mode'"'))
if !inlist(`"`run_mode'"', "fresh", "resume") {
    display as error "run_mode must be fresh or resume"
    exit 198
}

/* Locate the repository after the src/analysis reorganisation. */
local repo_root `"`requested_root'"'
if `"`repo_root'"' == "" {
    local repo_root `"`c(pwd)'"'
    capture confirm file `"`repo_root'/src/analysis/reporting_utils.do"'
    if _rc {
        local repo_root `"`c(pwd)'/.."'
        capture confirm file `"`repo_root'/src/analysis/reporting_utils.do"'
    }
    if _rc {
        local repo_root `"`c(pwd)'/../.."'
        capture confirm file `"`repo_root'/src/analysis/reporting_utils.do"'
    }
}
capture confirm file `"`repo_root'/src/analysis/reporting_utils.do"'
if _rc {
    display as error "Could not locate src/analysis/reporting_utils.do from `c(pwd)'"
    exit 601
}

local analysis_dir `"`repo_root'/src/analysis"'
local seed_file `"`repo_root'/assets/seed_list.txt"'
local data_dir `"`repo_root'/data/mvprobit"'
local raw_dir `"`data_dir'/raw"'
local output_dir `"`data_dir'/results"'
local log_dir `"`repo_root'/results/results_logs"'
local raw_data `"`raw_dir'/school.dta"'
local school_url "https://www.stata-press.com/data/r7/school.dta"
capture mkdir `"`raw_dir'"'
capture mkdir `"`output_dir'"'
capture mkdir `"`log_dir'"'
quietly do `"`analysis_dir'/reporting_utils.do"'

/* Requested production dimensions. */
local B 100
local m 100
local visualization_seeds 1000
local final_draws 150
local first_draws 2
local draw_step 2
local diagnostic_draws "2 150"
local diagnostic_draw_count 2
local draws_per_seed = (`final_draws' - `first_draws') / `draw_step' + 1
local algorithm_seed_count = max(`m', `visualization_seeds')
local required_seed_count = `algorithm_seed_count' + `B'
if `B' < 2 | `m' < 2 | `visualization_seeds' < 2 {
    display as error "S5 requires B>=2 and m>=2; the plot sweep needs >=2 seeds"
    exit 198
}

local crossed_checkpoint `"`output_dir'/mvprobit_crossed_scores_checkpoint.dta"'
local observed_checkpoint `"`output_dir'/mvprobit_visualization_checkpoint.dta"'
local bootstrap_indices_path `"`output_dir'/mvprobit_bootstrap_indices.dta"'
local observed_snapshot `"`output_dir'/mvprobit_observed_input.dta"'
local seed_plan_path `"`output_dir'/mvprobit_seed_plan.dta"'
local crossed_csv `"`output_dir'/mvprobit_crossed_bootstrap_scores.csv"'
local observed_csv `"`output_dir'/mvprobit_visualization_runs.csv"'
local compatibility_csv `"`data_dir'/results_school_total_draws150_total_seeds1000.csv"'
local diagnostics_csv `"`output_dir'/mvprobit_s5_diagnostics.csv"'
local observed_summary_csv `"`output_dir'/mvprobit_observed_seed_summary.csv"'
local log_path `"`log_dir'/mvprobit_s5_diagnostics.log"'
local pending_log `"`log_dir'/mvprobit_s5_diagnostics.pending.log"'
if `"`run_mode'"' == "fresh" {
    capture erase `"`crossed_checkpoint'"'
    capture erase `"`observed_checkpoint'"'
    capture erase `"`bootstrap_indices_path'"'
    capture erase `"`pending_log'"'
}

capture which mvprobit
if _rc {
    display as error "mvprobit is not installed or visible on the ado-path"
    exit 199
}
local mvprobit_ado `"`r(fn)'"'

/* Cache the public input so resumed runs do not need another download. */
capture confirm file `"`raw_data'"'
if _rc {
    display as text "Caching the Stata Press school dataset at `raw_data'"
    capture noisily copy `"`school_url'"' `"`raw_data'"', replace
    local copy_rc = _rc
    if `copy_rc' {
        display as error "Could not obtain `school_url'"
        exit `copy_rc'
    }
}
use `"`raw_data'"', clear
foreach variable in private vote years logptax loginc {
    capture confirm numeric variable `variable'
    if _rc {
        display as error "school.dta is missing numeric variable `variable'"
        exit 111
    }
}
quietly count
local n_observed = r(N)
if `n_observed' < 2 {
    display as error "school.dta must contain at least two observations"
    exit 2001
}
capture drop Source_Row
generate long Source_Row = _n
order Source_Row private vote years logptax loginc
tempfile observed_data
save `"`observed_data'"', replace
if `"`run_mode'"' == "fresh" {
    save `"`observed_snapshot'"', replace
}
else {
    capture confirm file `"`observed_snapshot'"'
    if _rc {
        display as error "Resume requires the saved observed-data provenance snapshot"
        exit 601
    }
    capture noisily cf Source_Row private vote years logptax loginc ///
        using `"`observed_snapshot'"', all
    local data_compare_rc = _rc
    if `data_compare_rc' {
        display as error "Current school.dta differs from the checkpoint provenance snapshot"
        exit `data_compare_rc'
    }
}

/* Disjoint algorithm-seed and external-bootstrap seed blocks. */
capture confirm file `"`seed_file'"'
if _rc {
    display as error "Seed list not found: `seed_file'"
    exit 601
}
import delimited using `"`seed_file'"', varnames(nonames) clear stringcols(_all)
capture confirm variable v1
if _rc {
    display as error "The seed list must contain one value per line"
    exit 459
}
rename v1 Seed_Text
replace Seed_Text = strtrim(Seed_Text)
drop if Seed_Text == ""
generate double Seed_Value = real(Seed_Text)
quietly count if missing(Seed_Value) | Seed_Value != floor(Seed_Value) | ///
    Seed_Value < 0 | Seed_Value > 2147483647
if r(N) {
    display as error "Seed list contains an invalid Stata seed"
    exit 459
}
duplicates tag Seed_Value, generate(Duplicate_Seed)
quietly count if Duplicate_Seed
if r(N) {
    display as error "Seed list values must be unique"
    exit 459
}
quietly count
if r(N) < `required_seed_count' {
    display as error "Seed list has `r(N)' entries; `required_seed_count' are required"
    exit 459
}
keep in 1/`required_seed_count'
generate long Seed_List_Position = _n
generate str12 Stage = cond(Seed_List_Position <= `algorithm_seed_count', ///
    "algorithm", "bootstrap")
generate int Seed_Vector = Seed_List_Position if Stage == "algorithm"
generate double Simulation_Seed = Seed_Value if Stage == "algorithm"
generate int Bootstrap_Replicate = ///
    Seed_List_Position - `algorithm_seed_count' if Stage == "bootstrap"
generate double Bootstrap_Seed = Seed_Value if Stage == "bootstrap"
format Seed_Value Simulation_Seed Bootstrap_Seed %21.0f
drop Seed_Text Duplicate_Seed
order Seed_List_Position Stage Seed_Vector Simulation_Seed ///
    Bootstrap_Replicate Bootstrap_Seed Seed_Value
save `"`seed_plan_path'"', replace
export delimited using `"`output_dir'/mvprobit_seed_plan.csv"', replace
tempfile algorithm_seed_plan bootstrap_seed_plan
preserve
keep if Stage == "algorithm" & Seed_Vector <= `visualization_seeds'
keep Seed_Vector Simulation_Seed
save `"`algorithm_seed_plan'"', replace
export delimited using `"`output_dir'/mvprobit_visualization_seed_plan.csv"', replace
keep if Seed_Vector <= `m'
export delimited using `"`output_dir'/mvprobit_s5_seed_plan.csv"', replace
restore
preserve
keep if Stage == "bootstrap"
keep Bootstrap_Replicate Bootstrap_Seed
save `"`bootstrap_seed_plan'"', replace
export delimited using `"`output_dir'/mvprobit_bootstrap_plan.csv"', replace
restore
capture frame drop mv_seed_plan
frame create mv_seed_plan
frame mv_seed_plan: use `"`seed_plan_path'"', clear

/* Materialise D_b*: source-row IDs make shared resampling auditable. */
local use_saved_bootstrap_indices 0
if `"`run_mode'"' == "resume" {
    capture confirm file `"`bootstrap_indices_path'"'
    if !_rc local use_saved_bootstrap_indices 1
}
if `use_saved_bootstrap_indices' {
    use `"`bootstrap_indices_path'"', clear
    capture isid Bootstrap_Replicate Draw_Position
    if _rc {
        display as error "Saved bootstrap index block has duplicate positions"
        exit 459
    }
    quietly count
    if r(N) != `B' * `n_observed' {
        display as error "Saved bootstrap index block has the wrong row count"
        exit 459
    }
    bysort Bootstrap_Replicate: assert _N == `n_observed'
    assert inrange(Bootstrap_Replicate, 1, `B')
    assert inrange(Draw_Position, 1, `n_observed')
    assert inrange(Source_Row, 1, `n_observed')
    forvalues b = 1/`B' {
        local expected_seed ""
        frame mv_seed_plan: quietly levelsof Bootstrap_Seed ///
            if Bootstrap_Replicate == `b', local(expected_seed) clean
        quietly count if Bootstrap_Replicate == `b' & ///
            Bootstrap_Seed != real(`"`expected_seed'"')
        if r(N) {
            display as error "Saved bootstrap seed mapping differs at row `b'"
            exit 459
        }
    }
}
else {
    tempfile bootstrap_indices_work
    tempname bootstrap_post
    postfile `bootstrap_post' int Bootstrap_Replicate long Draw_Position ///
        double Bootstrap_Seed long Source_Row using ///
        `"`bootstrap_indices_work'"', replace
    forvalues b = 1/`B' {
        local bootstrap_seed ""
        frame mv_seed_plan: quietly levelsof Bootstrap_Seed ///
            if Bootstrap_Replicate == `b', local(bootstrap_seed) clean
        if `"`bootstrap_seed'"' == "" {
            postclose `bootstrap_post'
            display as error "Missing bootstrap seed for replicate `b'"
            exit 459
        }
        use `"`observed_data'"', clear
        set seed `bootstrap_seed'
        bsample
        forvalues position = 1/`n_observed' {
            post `bootstrap_post' (`b') (`position') (`bootstrap_seed') ///
                (Source_Row[`position'])
        }
    }
    postclose `bootstrap_post'
    use `"`bootstrap_indices_work'"', clear
    isid Bootstrap_Replicate Draw_Position
    sort Bootstrap_Replicate Draw_Position
    save `"`bootstrap_indices_path'"', replace
}
export delimited using `"`output_dir'/mvprobit_bootstrap_indices.csv"', replace

/* Durable, row-complete crossed checkpoint. */
capture confirm file `"`crossed_checkpoint'"'
if _rc {
    clear
    set obs 0
    generate int Bootstrap_Replicate = .
    generate int Seed_Vector = .
    generate double Bootstrap_Seed = .
    generate double Simulation_Seed = .
    generate int draws = .
    generate double rho21 = .
    save `"`crossed_checkpoint'"', emptyok replace
}
use `"`crossed_checkpoint'"', clear
quietly count
if r(N) {
    isid Bootstrap_Replicate Seed_Vector draws
    assert inrange(Bootstrap_Replicate, 1, `B')
    assert inrange(Seed_Vector, 1, `m')
    assert inlist(draws, `first_draws', `final_draws')
    preserve
    rename Simulation_Seed Saved_Simulation_Seed
    merge m:1 Seed_Vector using `"`algorithm_seed_plan'"', ///
        keepusing(Simulation_Seed) assert(match) keep(match) nogen
    assert Saved_Simulation_Seed == Simulation_Seed
    rename Bootstrap_Seed Saved_Bootstrap_Seed
    merge m:1 Bootstrap_Replicate using `"`bootstrap_seed_plan'"', ///
        keepusing(Bootstrap_Seed) assert(match) keep(match) nogen
    assert Saved_Bootstrap_Seed == Bootstrap_Seed
    restore
}

tempname rho_scalar converged_scalar
forvalues b = 1/`B' {
    use `"`crossed_checkpoint'"', clear
    capture isid Bootstrap_Replicate Seed_Vector draws
    if _rc {
        display as error "Crossed checkpoint contains duplicate cells"
        exit 459
    }
    quietly count if Bootstrap_Replicate == `b'
    local existing_cells = r(N)
    if `existing_cells' == `m' * `diagnostic_draw_count' {
        display as text "Crossed bootstrap `b'/`B' already complete"
        continue
    }
    if `existing_cells' != 0 {
        display as error "Crossed checkpoint contains partial row `b'"
        exit 459
    }
    use `"`bootstrap_indices_path'"' if Bootstrap_Replicate == `b', clear
    merge m:1 Source_Row using `"`observed_data'"', ///
        keepusing(private vote years logptax loginc) keep(match) nogen
    sort Draw_Position
    quietly count
    if r(N) != `n_observed' {
        display as error "Bootstrap replicate `b' did not reconstruct correctly"
        exit 459
    }
    tempfile crossed_row
    tempname crossed_post
    postfile `crossed_post' int Bootstrap_Replicate int Seed_Vector ///
        double Bootstrap_Seed double Simulation_Seed int draws double rho21 ///
        using `"`crossed_row'"', replace
    local bootstrap_seed ""
    frame mv_seed_plan: quietly levelsof Bootstrap_Seed ///
        if Bootstrap_Replicate == `b', local(bootstrap_seed) clean
    forvalues j = 1/`m' {
        local simulation_seed ""
        frame mv_seed_plan: quietly levelsof Simulation_Seed ///
            if Seed_Vector == `j', local(simulation_seed) clean
        foreach diagnostic_draw of numlist `diagnostic_draws' {
            capture quietly mvprobit ///
                (private = years logptax loginc) ///
                (vote = years logptax loginc), ///
                seed(`simulation_seed') draws(`diagnostic_draw')
            local fit_rc = _rc
            if `fit_rc' {
                postclose `crossed_post'
                display as error "mvprobit failed in bootstrap `b', seed vector `j', draws=`diagnostic_draw' (rc=`fit_rc')"
                display as error "No fallback is substituted; diagnose it and resume"
                exit `fit_rc'
            }
            capture scalar `converged_scalar' = e(converged)
            if !_rc & scalar(`converged_scalar') != 1 {
                postclose `crossed_post'
                display as error "mvprobit did not converge in bootstrap `b', seed vector `j', draws=`diagnostic_draw'"
                display as error "No fallback is substituted; diagnose it and resume"
                exit 430
            }
            scalar `rho_scalar' = e(rho21)
            if missing(scalar(`rho_scalar')) | abs(scalar(`rho_scalar')) > 1 {
                postclose `crossed_post'
                display as error "Invalid e(rho21) in bootstrap `b', seed vector `j', draws=`diagnostic_draw'"
                exit 498
            }
            post `crossed_post' (`b') (`j') (`bootstrap_seed') ///
                (`simulation_seed') (`diagnostic_draw') (scalar(`rho_scalar'))
        }
    }
    postclose `crossed_post'
    use `"`crossed_checkpoint'"', clear
    append using `"`crossed_row'"'
    isid Bootstrap_Replicate Seed_Vector draws
    sort Bootstrap_Replicate Seed_Vector
    save `"`crossed_checkpoint'"', replace
    display as text "Completed crossed bootstrap `b'/`B' at `c(current_time)'"
}

use `"`crossed_checkpoint'"', clear
isid Bootstrap_Replicate Seed_Vector draws
quietly count
if r(N) != `B' * `m' * `diagnostic_draw_count' {
    display as error "Crossed grid is incomplete: found `r(N)' cells"
    exit 459
}
bysort Bootstrap_Replicate: assert _N == `m' * `diagnostic_draw_count'
bysort Bootstrap_Replicate Seed_Vector: assert _N == `diagnostic_draw_count'
bysort draws: assert _N == `B' * `m'
assert inlist(draws, `first_draws', `final_draws')
assert !missing(rho21)
assert inrange(rho21, -1, 1)
sort Bootstrap_Replicate Seed_Vector
save `"`output_dir'/mvprobit_crossed_bootstrap_scores.dta"', replace
export delimited using `"`crossed_csv'"', replace

/* Persist unrounded S5 quantities for both reported draw configurations. */
tempfile diagnostics_work
tempname diagnostics_post
postfile `diagnostics_post' str40 Model str80 Estimand int draws ///
    int B_Bootstrap_Resamples int M_Seed_Vectors ///
    double Seed_Averaged_Estimate double Data_Uncertainty_SD ///
    double Data_Variance double Between_Seed_SD ///
    double Seed_Variance_Nonnegative double Seed_Variance_Raw ///
    byte Boundary_Hit double Relative_Importance double Algorithmic_Share ///
    double Total_Order_Share double Data_Main_Variance ///
    double Data_Seed_Interaction double MS_Seed double MS_Data ///
    double MS_Interaction using `"`diagnostics_work'"', replace
foreach diagnostic_draw of numlist `diagnostic_draws' {
    use `"`crossed_checkpoint'"' if draws == `diagnostic_draw', clear
    s5_crossed_diagnostics, score(rho21) bootstrap(Bootstrap_Replicate) ///
        seed(Seed_Vector)
    local theta_bar = r(seed_averaged_estimate)
    local sigma_d_squared = r(data_variance)
    local sd_theta_d = r(data_uncertainty_sd)
    local sigma_s_raw = r(between_seed_variance_raw)
    local sigma_s_report = r(between_seed_variance)
    local boundary_hit = r(variance_component_boundary_hit)
    local sd_theta_s = r(between_seed_variability_sd)
    local relative_r = r(relative_importance)
    local rho_s = r(algorithmic_variance_share)
    local rho_t_s = r(total_order_algorithmic_share)
    local v_d_hat = r(data_main_effect_variance)
    local v_ds_hat = r(data_seed_interaction_variance)
    local ms_s = r(seed_mean_square)
    local ms_d = r(data_mean_square)
    local ms_int = r(interaction_mean_square)
    post `diagnostics_post' ("MVProbit") ///
        ("rho21 at `diagnostic_draw' simulation draws") (`diagnostic_draw') ///
        (`B') (`m') (`theta_bar') (`sd_theta_d') (`sigma_d_squared') ///
        (`sd_theta_s') (`sigma_s_report') (`sigma_s_raw') (`boundary_hit') ///
        (`relative_r') (`rho_s') (`rho_t_s') (`v_d_hat') (`v_ds_hat') ///
        (`ms_s') (`ms_d') (`ms_int')
}
postclose `diagnostics_post'
use `"`diagnostics_work'"', clear
sort draws
export delimited using `"`diagnostics_csv'"', replace

/* Durable plot-sweep checkpoint: one complete seed block per save. */
capture confirm file `"`observed_checkpoint'"'
if _rc {
    clear
    set obs 0
    generate int Seed_Vector = .
    generate double Simulation_Seed = .
    generate int draws = .
    generate double rho21 = .
    generate int fit_return_code = .
    save `"`observed_checkpoint'"', emptyok replace
}
use `"`observed_checkpoint'"', clear
quietly count
if r(N) {
    isid Seed_Vector draws
    assert inrange(Seed_Vector, 1, `visualization_seeds')
    assert inrange(draws, `first_draws', `final_draws')
    assert mod(draws, `draw_step') == 0
    preserve
    rename Simulation_Seed Saved_Simulation_Seed
    merge m:1 Seed_Vector using `"`algorithm_seed_plan'"', ///
        keepusing(Simulation_Seed) assert(match) keep(match) nogen
    assert Saved_Simulation_Seed == Simulation_Seed
    restore
}
forvalues j = 1/`visualization_seeds' {
    use `"`observed_checkpoint'"', clear
    capture isid Seed_Vector draws
    if _rc {
        display as error "Plot checkpoint contains duplicate seed/draw pairs"
        exit 459
    }
    quietly count if Seed_Vector == `j'
    local existing_draws = r(N)
    if `existing_draws' == `draws_per_seed' continue
    if `existing_draws' != 0 {
        display as error "Plot checkpoint contains partial seed vector `j'"
        exit 459
    }
    local simulation_seed ""
    frame mv_seed_plan: quietly levelsof Simulation_Seed ///
        if Seed_Vector == `j', local(simulation_seed) clean
    use `"`observed_data'"', clear
    tempfile observed_seed_row
    tempname observed_post
    postfile `observed_post' int Seed_Vector double Simulation_Seed int draws ///
        double rho21 int fit_return_code using `"`observed_seed_row'"', replace
    forvalues draw = `first_draws'(`draw_step')`final_draws' {
        capture quietly mvprobit ///
            (private = years logptax loginc) ///
            (vote = years logptax loginc), ///
            seed(`simulation_seed') draws(`draw')
        local fit_rc = _rc
        if `fit_rc' {
            post `observed_post' (`j') (`simulation_seed') (`draw') (.) (`fit_rc')
            continue
        }
        capture scalar `converged_scalar' = e(converged)
        if !_rc & scalar(`converged_scalar') != 1 {
            post `observed_post' (`j') (`simulation_seed') (`draw') (.) (430)
            continue
        }
        scalar `rho_scalar' = e(rho21)
        if missing(scalar(`rho_scalar')) | abs(scalar(`rho_scalar')) > 1 {
            post `observed_post' (`j') (`simulation_seed') (`draw') (.) (498)
            continue
        }
        post `observed_post' (`j') (`simulation_seed') (`draw') ///
            (scalar(`rho_scalar')) (0)
    }
    postclose `observed_post'
    use `"`observed_checkpoint'"', clear
    append using `"`observed_seed_row'"'
    isid Seed_Vector draws
    sort Seed_Vector draws
    save `"`observed_checkpoint'"', replace
    if mod(`j', 10) == 0 | `j' == `visualization_seeds' {
        display as text "Completed observed-data seed `j'/`visualization_seeds' at `c(current_time)'"
    }
}

use `"`observed_checkpoint'"', clear
isid Seed_Vector draws
quietly count
if r(N) != `visualization_seeds' * `draws_per_seed' {
    display as error "Plot sweep has `r(N)' rows instead of 75,000"
    exit 459
}
bysort Seed_Vector: assert _N == `draws_per_seed'
assert inrange(draws, `first_draws', `final_draws')
assert mod(draws, `draw_step') == 0
sort Seed_Vector draws
save `"`output_dir'/mvprobit_visualization_runs.dta"', replace
export delimited using `"`observed_csv'"', replace
quietly count if fit_return_code != 0 | missing(rho21)
local visualization_failed_cells = r(N)
foreach diagnostic_draw of numlist `diagnostic_draws' {
    quietly count if draws == `diagnostic_draw' & ///
        (fit_return_code != 0 | missing(rho21))
    if r(N) {
        display as error "A `diagnostic_draw'-draw plot fit failed; its conditional summary is incomplete"
        exit 459
    }
}

/* Exact schema/path consumed by helper_figure_plotters.py. */
preserve
generate double seed = Simulation_Seed
format seed %21.0f
keep seed draws rho21
order seed draws rho21
export delimited using `"`compatibility_csv'"', replace
restore

tempfile observed_summary_work
tempname observed_summary_post
postfile `observed_summary_post' str80 Estimand int draws ///
    int N_Seed_Vectors double Mean double SD_Conditional_On_D_Observed ///
    double Minimum double Maximum long Failed_Seed_Draw_Attempts using ///
    `"`observed_summary_work'"', replace
foreach diagnostic_draw of numlist `diagnostic_draws' {
    preserve
    keep if draws == `diagnostic_draw'
    s5_observed_summary, score(rho21)
    local observed_n_`diagnostic_draw' = r(n_seed_vectors)
    local observed_mean_`diagnostic_draw' = r(observed_data_seed_average)
    local observed_sd_`diagnostic_draw' = r(observed_between_seed_sd)
    local observed_min_`diagnostic_draw' = r(minimum)
    local observed_max_`diagnostic_draw' = r(maximum)
    restore
    post `observed_summary_post' ///
        ("rho21 at `diagnostic_draw' draws, conditional on observed data") ///
        (`diagnostic_draw') (`observed_n_`diagnostic_draw'') ///
        (`observed_mean_`diagnostic_draw'') (`observed_sd_`diagnostic_draw'') ///
        (`observed_min_`diagnostic_draw'') (`observed_max_`diagnostic_draw'') ///
        (`visualization_failed_cells')
}
postclose `observed_summary_post'
use `"`observed_summary_work'"', clear
sort draws
export delimited using `"`observed_summary_csv'"', replace

/* Replace the canonical metrics log only after both experiments complete. */
foreach diagnostic_draw of numlist `diagnostic_draws' {
    use `"`crossed_checkpoint'"' if draws == `diagnostic_draw', clear
    local report_mode ""
    if `diagnostic_draw' == `first_draws' local report_mode "replace"
    s5_write_report using `"`pending_log'"', score(rho21) ///
        bootstrap(Bootstrap_Replicate) seed(Seed_Vector) ///
        estimand("MVProbit rho21 at `diagnostic_draw' simulation draws") ///
        `report_mode' withinrep("1") ///
        prng("Stata `c(rng)' bootstrap RNG; mvprobit seed() GHK simulator RNG") ///
        variation("one simulation-seed component varied jointly at all stochastic mvprobit stages") ///
        details("draws=`diagnostic_draw'; data_source=`school_url'; data_snapshot=`observed_snapshot'; bootstrap_unit=school.dta observation; bootstrap_size=`n_observed'; model=(private years logptax loginc) (vote years logptax loginc); convergence_policy=no fallback and e(converged) checked when exposed; Stata=`c(stata_version)'; Stata_flavor=`c(flavor)'; processors=`c(processors)'; mvprobit_ado=`mvprobit_ado'")
}
foreach diagnostic_draw of numlist `diagnostic_draws' {
    local log_observed_n = `observed_n_`diagnostic_draw''
    local log_observed_mean = `observed_mean_`diagnostic_draw''
    local log_observed_sd = `observed_sd_`diagnostic_draw''
    local log_observed_min = `observed_min_`diagnostic_draw''
    local log_observed_max = `observed_max_`diagnostic_draw''
    s5_log_line using `"`pending_log'"', ///
        text("Observed-data seed sweep (no bootstrap): MVProbit rho21 at `diagnostic_draw' simulation draws")
    s5_log_line using `"`pending_log'"', text("n_seed_vectors=`log_observed_n'")
    s5_log_line using `"`pending_log'"', ///
        text("observed_data_seed_average=`log_observed_mean'")
    s5_log_line using `"`pending_log'"', ///
        text("observed_between_seed_sd_conditional_on_D_obs=`log_observed_sd'")
    s5_log_line using `"`pending_log'"', text("minimum=`log_observed_min'")
    s5_log_line using `"`pending_log'"', text("maximum=`log_observed_max'")
    s5_log_line using `"`pending_log'"', ///
        text("failed_seed_draw_attempts_all_draw_counts=`visualization_failed_cells'")
    s5_log_line using `"`pending_log'"', ///
        text("interpretation=Conditional variability on D_obs; it estimates V_S + V_DS.")
    s5_log_line using `"`pending_log'"', ///
        text("not_reported=Data uncertainty, r, and rho_S are not computed from this sweep.")
}
s5_validate_log using `"`pending_log'"', ///
    estimands("MVProbit rho21 at 2 simulation draws|MVProbit rho21 at 150 simulation draws")
copy `"`pending_log'"' `"`log_path'"', replace
capture erase `"`pending_log'"'

display as result "MVProbit S5 diagnostics complete"
display as result "Metrics log: `log_path'"
display as result "Crossed scores: `crossed_csv'"
display as result "Plot input: `compatibility_csv'"
