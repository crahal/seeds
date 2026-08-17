"""Static contracts for the Stata S5 rewrite.

These tests deliberately read source text only.  They never invoke Stata or
execute either .do file, matching the requested no-run validation policy.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "src" / "analysis" / "build_mvprobit.do"
UTILS = REPO_ROOT / "src" / "analysis" / "reporting_utils.do"


def _compact(path: Path) -> str:
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8").replace("///", " "))


def test_stata_reporting_utils_exposes_s5_contract() -> None:
    text = UTILS.read_text(encoding="utf-8")
    for program in (
        "s5_crossed_diagnostics",
        "s5_observed_summary",
        "s5_write_report",
        "s5_log_line",
        "s5_validate_log",
    ):
        assert f"program define {program}" in text

    for heading in (
        "1. Seed-averaged estimate",
        "2. Data uncertainty",
        "3. Bias-corrected between-seed variability",
        "4. Relative importance of algorithmic randomness",
        "5. Algorithmic variance share",
        "6. Computational details",
    ):
        assert heading in text


def test_stata_s5_formulas_match_equations_25_to_27() -> None:
    text = _compact(UTILS)
    required_fragments = (
        "bysort `seed': egen double `seed_sd' = sd(`score')",
        "local data_variance = r(mean)",
        "local seed_ms = `B' * r(Var)",
        "local data_ms = `m' * r(Var)",
        "local interaction_ms = r(min) / ((`B' - 1) * (`m' - 1))",
        "local seed_variance_raw = (`seed_ms' - `interaction_ms') / `B'",
        "local data_main_variance = (`data_ms' - `interaction_ms') / `m'",
        "local seed_variance_report = max(0, `seed_variance_raw')",
        "(`seed_variance_report' + `interaction_ms') / `denominator'",
    )
    for fragment in required_fragments:
        assert fragment in text
    assert "summarize `seed_mean' if `tag_seed', meanonly" not in text
    assert "summarize `bootstrap_mean' if `tag_bootstrap', meanonly" not in text
    assert "return scalar total_order_algorithmic_share" in text
    assert "local block_start" in text
    assert "strpos(`\"`block'\"', `\"`heading'\"')" in text

    # Stata identifiers, including r()-return names, are limited to 32 chars.
    return_names = re.findall(r"return scalar\s+([A-Za-z_][A-Za-z0-9_]*)", text)
    assert return_names
    assert max(map(len, return_names)) <= 32


def test_mvprobit_requested_dimensions_and_crossing() -> None:
    text = _compact(BUILDER)
    assert "local B 100" in text
    assert "local m 100" in text
    assert "local visualization_seeds 1000" in text
    assert 'local diagnostic_draws "2 150"' in text

    bootstrap_load = text.index(
        'use `"`bootstrap_indices_path\'"\' if Bootstrap_Replicate == `b\''
    )
    seed_loop = text.index("forvalues j = 1/`m'", bootstrap_load)
    draw_loop = text.index(
        "foreach diagnostic_draw of numlist `diagnostic_draws'", seed_loop
    )
    score_post = text.index("post `crossed_post'", draw_loop)
    assert bootstrap_load < seed_loop < draw_loop < score_post
    assert "isid Bootstrap_Replicate Seed_Vector draws" in text
    assert "bysort Bootstrap_Replicate Seed_Vector: assert _N ==" in text
    assert "No fallback is substituted" in text
    assert "Saved_Simulation_Seed == Simulation_Seed" in text
    assert "Saved_Bootstrap_Seed == Bootstrap_Seed" in text
    assert "mvprobit_observed_input.dta" in text


def test_mvprobit_visualization_and_log_compatibility() -> None:
    text = BUILDER.read_text(encoding="utf-8")
    assert "results_school_total_draws150_total_seeds1000.csv" in text
    assert "forvalues draw = `first_draws'(`draw_step')`final_draws'" in text
    assert "keep seed draws rho21" in text
    assert "continue, break" not in text
    assert "fit_return_code" in text

    assert "results/results_logs" in text
    assert "mvprobit_s5_diagnostics.log" in text
    assert "mvprobit_s5_diagnostics.pending.log" in text
    assert "s5_write_report" in text
    assert (
        'estimands("MVProbit rho21 at 2 simulation draws|MVProbit rho21 at '
        '150 simulation draws")'
        in _compact(BUILDER)
    )


def test_builder_generated_variable_names_fit_stata_limit() -> None:
    text = BUILDER.read_text(encoding="utf-8")
    generated_names = re.findall(
        r"\bgenerate\s+(?:(?:byte|int|long|float|double|str\d+)\s+)?"
        r"([A-Za-z_][A-Za-z0-9_]*)",
        text,
    )
    assert generated_names
    assert max(map(len, generated_names)) <= 32
