"""
Causal-forest seed-variability simulation (I/M/E only), CSV-only, parallel.

Design (researcher-like)
-----------------------
• Allocation is drawn ONCE and kept fixed.
• Imputation uses MICE-style chained equations with posterior sampling:
  - Fit the imputer on TRAINING covariates only (no test leakage).
  - Transform the test covariates with that fitted imputer.
  - Repeat for M imputations and average test CATEs; record MI diagnostics.
• CausalForestDML with cross-fitting for nuisance models; moderate forest size.
• Seed pools come from ../assets/seed_list.txt (one int per line). The "fixed" seeds
  are the FIRST element of each pool so single-factor schemes are nested in vary_all.
• Dispersion metric is identical across schemes: ate_sd = SD of test ATE across runs.
• Outputs: results_long.csv, summary_by_scheme.csv, metadata.json (no plotting).

Dependencies
------------
pip install numpy pandas scikit-learn econml tqdm joblib tqdm_joblib
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, KFold
from econml.dml import CausalForestDML

from joblib import Parallel, delayed
from tqdm.auto import tqdm
try:
    from tqdm_joblib import tqdm_joblib
    _HAS_TQDM_JOBLIB = True
except Exception:
    _HAS_TQDM_JOBLIB = False


# -----------------------------
# Utilities
# -----------------------------

def get_seed_list() -> List[int]:
    """Read seeds from ../assets/seed_list.txt (one integer per line)."""
    seed_list_path = os.path.join(os.getcwd(), '..', 'assets', 'seed_list.txt')
    with open(seed_list_path) as f:
        return [int(line.rstrip('\n')) for line in f]

def _round_up_multiple(x: int, k: int) -> int:
    """Round x up to the nearest multiple of k."""
    return int(np.ceil(x / k) * k)


# -----------------------------
# 0) Fixed baseline DGP (with ONE-TIME allocation)
# -----------------------------

@dataclass
class BaselineData:
    """Fixed 'world': covariates (with MAR mask), potential outcomes, and ONE fixed allocation."""
    X_obs: np.ndarray
    X_true: np.ndarray
    Y0: np.ndarray
    Y1: np.ndarray
    T_fixed: np.ndarray
    Y_obs: np.ndarray


def generate_baseline_data(
    N: int = 600,
    d: int = 16,
    base_seed: int = 20240101,
    tau: float = 0.3,
    miss_rate: float = 0.50,   # moderate missingness typical of applied work
    alloc_seed: int = 11,
) -> BaselineData:
    """Generate one fixed dataset and ONE fixed allocation T (and corresponding observed Y)."""
    rng = np.random.default_rng(base_seed)

    # Correlated Gaussian X
    cov = 0.3 * np.ones((d, d)) + 0.7 * np.eye(d)
    L = np.linalg.cholesky(cov)
    X = rng.standard_normal((N, d)) @ L.T

    # Potential outcomes via logistic link
    beta = np.linspace(1.2, 0.4, d)
    lin = -0.5 + X @ beta
    p0 = 1.0 / (1.0 + np.exp(-lin))
    p1 = 1.0 / (1.0 + np.exp(-(lin + tau)))
    Y0 = (rng.random(N) < p0).astype(int)
    Y1 = (rng.random(N) < p1).astype(int)

    # ONE-TIME random allocation
    rngA = np.random.default_rng(alloc_seed)
    T_fixed = rngA.integers(0, 2, size=N).astype(int)
    Y_obs = np.where(T_fixed == 1, Y1, Y0)

    # Fixed MAR mask on X
    logits = (np.abs(X) - np.mean(np.abs(X), axis=0))
    logits /= (np.std(logits, axis=0, ddof=1) + 1e-9)
    logits -= np.quantile(logits, 1 - miss_rate, axis=0)
    pmiss = 1.0 / (1.0 + np.exp(-logits))
    M = (rng.random(X.shape) < pmiss)
    X_obs = X.copy(); X_obs[M] = np.nan

    return BaselineData(X_obs=X_obs, X_true=X, Y0=Y0, Y1=Y1, T_fixed=T_fixed, Y_obs=Y_obs)


# -----------------------------
# 1) One causal-forest run (I, M, E vary; A fixed once)
# -----------------------------

def run_cf_once(
    base: BaselineData,
    seed_imp: int,
    seed_model: int,
    seed_eval: int,
    # Researcher-like defaults
    M_imputations: int = 1,        # conventional MI count
    imputer_max_iter: int = 10,    # default-ish MICE iterations
    imputer_tol: float = 1e-3,     # reasonable tolerance
    test_size: float = 0.25,       # 75/25 train/test split
    cv_folds: int = 3,             # cross-fitting folds for nuisance models
    n_estimators: int = 200,       # moderate forest
    subforest_size: int = 4,       # econml: n_estimators must be divisible by this
    min_samples_leaf: int = 5,     # avoid ultra-sparse leaves
    max_depth: Optional[int] = None,
    max_features: Optional[float] = 0.5,  # ~sqrt(d)/d heuristic as a fraction
) -> Dict[str, float]:
    """
    One pipeline run: (E) outer split, (I) MI (no artificial jitter), (M) causal forest.
    Allocation A is fixed once globally in `base`.
    Returns test-population summaries + MI diagnostics.
    """
    # Fixed observed T, Y from the one-time allocation
    T = base.T_fixed
    Y = base.Y_obs

    # (E) evaluation split (stratify on Y)
    X_tr_obs, X_te_obs, T_tr, T_te, Y_tr, Y_te = train_test_split(
        base.X_obs, T, Y, test_size=test_size, stratify=Y, random_state=seed_eval
    )
    cv_kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed_eval)

    # (I)+(M): MI + forest, average CATEs across imputations
    cate_preds: List[np.ndarray] = []
    ates_m: List[float] = []

    for m in range(M_imputations):
        # Fit imputer on TRAINING covariates; transform both train and test
        imp = IterativeImputer(
            sample_posterior=True,              # proper MI draw
            random_state=seed_imp + 1000 * m,   # different draw per m
            max_iter=imputer_max_iter, tol=imputer_tol,
        )
        X_tr = imp.fit_transform(X_tr_obs)
        X_te = imp.transform(X_te_obs)

        # econml divisibility constraint
        n_est = int(n_estimators)
        sfs = int(subforest_size)
        if n_est % sfs != 0:
            n_est = _round_up_multiple(n_est, sfs)

        # (M) causal forest (honest GRF via econml)
        cf = CausalForestDML(
            n_estimators=n_est,
            subforest_size=sfs,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_features=max_features,
            discrete_treatment=True,
            cv=cv_kf,                 # cross-fitting for nuisance functions
            random_state=seed_model,
        )
        cf.fit(Y_tr, T_tr, X=X_tr)
        tau_te = cf.effect(X_te)  # shape: (n_test,)
        cate_preds.append(tau_te)
        ates_m.append(float(np.mean(tau_te)))

    # Pool across imputations (average CATEs across M imputations)
    tau_te_bar = np.mean(np.column_stack(cate_preds), axis=1)

    # Test ATE and heterogeneity summaries
    ate_te = float(np.mean(tau_te_bar))
    cate_sd = float(np.std(tau_te_bar, ddof=1))
    cate_iqr = float(np.subtract(*np.quantile(tau_te_bar, [0.75, 0.25])))

    # MI diagnostics (between-imputation variability of test ATE)
    M_used = len(ates_m)
    if M_used > 1:
        mi_sd = float(np.std(ates_m, ddof=1))
        mi_iqr = float(np.subtract(*np.quantile(ates_m, [0.75, 0.25])))
        mi_B = float(np.var(ates_m, ddof=1))
    else:
        mi_sd = mi_iqr = mi_B = 0.0

    return {
        "ate_hat": ate_te,
        "cate_sd_test": cate_sd,
        "cate_iqr_test": cate_iqr,
        "mi_ate_sd": mi_sd,
        "mi_ate_iqr": mi_iqr,
        "mi_B": mi_B,
        "M_imputations": int(M_used),
        "n_test": int(len(tau_te_bar)),
    }


# -----------------------------
# 2) Seed pools & schemes (A removed)
# -----------------------------

ALLOWED_SCHEMES = {
    "baseline_fixed",
    "vary_imp",
    "vary_model",
    "vary_eval",
    "vary_all",
}

def _validate_schemes(schemes: List[str]) -> List[str]:
    bad = sorted(set(schemes) - ALLOWED_SCHEMES)
    if bad:
        raise ValueError(f"Unknown/disabled scheme(s): {bad}. Allowed: {sorted(ALLOWED_SCHEMES)}")
    return schemes


@dataclass
class SeedPools:
    imp: np.ndarray
    model: np.ndarray
    eval: np.ndarray


def make_seed_pools(
    R_imp: int, R_model: int, R_eval: int, *, use_seed_file: bool = True, master_seed: int = 7
) -> SeedPools:
    """
    Build seed pools. If `use_seed_file=True`, consume integers from
    ../assets/seed_list.txt in order:
        imp = first R_imp
        model = next R_model
        eval  = next R_eval
    Otherwise, draw from an RNG initialized with `master_seed`.
    """
    if use_seed_file:
        seeds = np.array(get_seed_list(), dtype=np.int64)
        need = R_imp + R_model + R_eval
        if len(seeds) < need:
            raise ValueError(f"Seed file too short: need {need}, found {len(seeds)}.")
        imp = seeds[:R_imp]
        model = seeds[R_imp:R_imp + R_model]
        eval_ = seeds[R_imp + R_model:R_imp + R_model + R_eval]
    else:
        rng = np.random.default_rng(master_seed)
        imp = rng.integers(1, 2**31 - 1, size=R_imp, dtype=np.int64)
        model = rng.integers(1, 2**31 - 1, size=R_model, dtype=np.int64)
        eval_ = rng.integers(1, 2**31 - 1, size=R_eval, dtype=np.int64)
    return SeedPools(imp=imp, model=model, eval=eval_)


def run_scheme_parallel(
    base: BaselineData,
    scheme: str,
    pools: SeedPools,
    R_imp: int,
    R_model: int,
    R_eval: int,
    fixed_seeds: Tuple[int, int, int],  # (imp, model, eval) == first of each pool
    # execution
    n_jobs: int,
    progress: bool,
    master_seed: int,
    vary_all_max_combos: Optional[int],
    # knobs passed through:
    M_imputations: int,
    imputer_max_iter: int,
    imputer_tol: float,
    test_size: float,
    cv_folds: int,
    n_estimators: int,
    subforest_size: int,
    min_samples_leaf: int,
    max_depth: Optional[int],
    max_features: Optional[float],
) -> pd.DataFrame:
    """
    Parallel execution:
      - vary_imp: vary pools.imp[i], hold model/eval at pools.model[0], pools.eval[0]
      - vary_model: vary pools.model[j], hold imp/eval at pools.imp[0], pools.eval[0]
      - vary_eval: vary pools.eval[k], hold imp/model at pools.imp[0], pools.model[0]
      - baseline_fixed: single run with (pools.imp[0], pools.model[0], pools.eval[0])
      - vary_all: Cartesian product of the three pools (optionally capped)
    """
    si0, sm0, se0 = fixed_seeds  # by construction: pools.imp[0], pools.model[0], pools.eval[0]

    def _worker(si: int, sm: int, se: int, idx_i: int, idx_m: int, idx_e: int) -> Dict[str, float]:
        out = run_cf_once(
            base,
            seed_imp=si, seed_model=sm, seed_eval=se,
            M_imputations=M_imputations,
            imputer_max_iter=imputer_max_iter, imputer_tol=imputer_tol,
            test_size=test_size, cv_folds=cv_folds,
            n_estimators=n_estimators, subforest_size=subforest_size,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth, max_features=max_features,
        )
        return {
            "scheme": scheme,
            "seed_imp": int(si), "seed_model": int(sm), "seed_eval": int(se),
            "idx_imp": idx_i, "idx_model": idx_m, "idx_eval": idx_e,
            **out
        }

    # Build job list
    if scheme == "baseline_fixed":
        jobs = [ (si0, sm0, se0, 0, 0, 0) ]
    elif scheme == "vary_imp":
        jobs = [ (int(pools.imp[i]), sm0, se0, i, 0, 0) for i in range(R_imp) ]
    elif scheme == "vary_model":
        jobs = [ (si0, int(pools.model[j]), se0, 0, j, 0) for j in range(R_model) ]
    elif scheme == "vary_eval":
        jobs = [ (si0, sm0, int(pools.eval[k]), 0, 0, k) for k in range(R_eval) ]
    elif scheme == "vary_all":
        total = R_imp * R_model * R_eval
        if (vary_all_max_combos is not None) and (total > vary_all_max_combos):
            rng = np.random.default_rng(master_seed + 12345)
            m = vary_all_max_combos
            lin_idx = rng.choice(total, size=m, replace=False)
            RM = R_model * R_eval
            jobs = []
            for u in lin_idx:
                i = int(u // RM); rem = int(u % RM)
                j = int(rem // R_eval); k = int(rem % R_eval)
                jobs.append((int(pools.imp[i]), int(pools.model[j]), int(pools.eval[k]), i, j, k))
        else:
            jobs = [
                (int(pools.imp[i]), int(pools.model[j]), int(pools.eval[k]), i, j, k)
                for i in range(R_imp) for j in range(R_model) for k in range(R_eval)
            ]
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    # Parallel execution
    iterator = (delayed(_worker)(si, sm, se, idx_i, idx_m, idx_e) for (si, sm, se, idx_i, idx_m, idx_e) in jobs)
    desc = f"Runs — {scheme}"
    if progress and _HAS_TQDM_JOBLIB:
        with tqdm_joblib(tqdm(total=len(jobs), desc=desc, leave=False)):
            rows = Parallel(n_jobs=n_jobs, prefer="processes")(iterator)
    else:
        rows = Parallel(n_jobs=n_jobs, prefer="processes")(iterator)
        if progress and not _HAS_TQDM_JOBLIB:
            _ = list(tqdm(range(len(jobs)), desc=desc, leave=False))

    return pd.DataFrame(rows)


# -----------------------------
# 3) Aggregation, I/O (CSV-only)
# -----------------------------

def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results by scheme with the same SD metric (across-run SD),
    include ATE min/max, and append a row aggregating ATE over all rows
    EXCLUDING the 'vary_all' scheme.
    """
    per_scheme = (results
                  .groupby("scheme", as_index=False)
                  .agg(ate_mean=("ate_hat", "mean"),
                       ate_sd=("ate_hat", "std"),
                       ate_min=("ate_hat", "min"),
                       ate_max=("ate_hat", "max"),
                       cate_sd_mean=("cate_sd_test", "mean"),
                       cate_iqr_mean=("cate_iqr_test", "mean"),
                       mi_ate_sd_mean=("mi_ate_sd", "mean"),
                       mi_ate_iqr_mean=("mi_ate_iqr", "mean"),
                       n=("ate_hat", "size"))
                  .sort_values("ate_sd", ascending=False))

    # Overall row excluding vary_all (ATE stats only; other cols set to NaN)
    mask = results["scheme"] != "vary_all"
    overall = {
        "scheme": "ALL_EXCL_vary_all",
        "ate_mean": float(results.loc[mask, "ate_hat"].mean()),
        "ate_sd": float(results.loc[mask, "ate_hat"].std(ddof=1)),
        "ate_min": float(results.loc[mask, "ate_hat"].min()),
        "ate_max": float(results.loc[mask, "ate_hat"].max()),
        "cate_sd_mean": np.nan,
        "cate_iqr_mean": np.nan,
        "mi_ate_sd_mean": np.nan,
        "mi_ate_iqr_mean": np.nan,
        "n": int(mask.sum()),
    }
    per_scheme = pd.concat([per_scheme, pd.DataFrame([overall])], ignore_index=True)
    return per_scheme


def save_outputs_csv_only(
    results: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    config_meta: Dict[str, object],
) -> None:
    """Write CSVs + a tiny metadata.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "results_long.csv", index=False)
    summary_df.to_csv(output_dir / "summary_by_scheme.csv", index=False)
    meta = {
        "n_rows": int(results.shape[0]),
        "schemes": sorted(results["scheme"].unique().tolist()),
        "columns": results.columns.tolist(),
        **config_meta,
    }
    pd.Series(meta, dtype=object).to_json(output_dir / "metadata.json", indent=2)


# -----------------------------
# 4) Main
# -----------------------------

def main(
    # Fixed world (allocation fixed)
    N: int = 600,
    d: int = 16,
    miss_rate: float = 0.50,
    alloc_seed: int = 42,
    tau: float = 0.1,

    # single-factor run counts:
    R_imp: int = 100,
    R_model: int = 100,
    R_eval: int = 100,

    # cap for vary_all Cartesian product (set None to disable capping):
    vary_all_max_combos: Optional[int] = None,

    # execution
    use_seed_file: bool = True,
    master_seed: int = 42,          # used only if use_seed_file=False or for vary_all sampling
    M_imputations: int = 1,
    n_jobs: int = -1,
    progress: bool = True,
    output_dir: str = "../data/compound/CF_sim_IME_only_exploded",
    clean_output: bool = True,

    # CF / MI knobs (researcher-like defaults)
    imputer_max_iter: int = 10,
    imputer_tol: float = 1e-3,
    test_size: float = 0.25,
    cv_folds: int = 3,
    n_estimators: int = 200,
    subforest_size: int = 4,
    min_samples_leaf: int = 5,
    max_depth: Optional[int] = None,
    max_features: Optional[float] = 0.5,

    schemes: Optional[List[str]] = None,
) -> None:
    """
    Run all schemes with the SAME SD metric (across-run SD), allocation fixed once.
    Seed pools come from `get_seed_list()` (or RNG fallback). Fixed seeds are the FIRST
    elements of each pool so single-factor schemes are nested in `vary_all`.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    base = generate_baseline_data(
        N=N, d=d, base_seed=20240101, tau=tau, miss_rate=miss_rate, alloc_seed=alloc_seed
    )

    if schemes is None:
        schemes = ["baseline_fixed","vary_imp","vary_model","vary_eval","vary_all"]
    schemes = _validate_schemes(schemes)

    # Build seed pools ONCE so vary_all can "explode" them.
    pools = make_seed_pools(
        R_imp=R_imp, R_model=R_model, R_eval=R_eval,
        use_seed_file=use_seed_file, master_seed=master_seed
    )

    # Fixed seeds are the FIRST elements of each pool (nesting guarantee).
    fixed_seeds = (int(pools.imp[0]), int(pools.model[0]), int(pools.eval[0]))

    outdir = Path(output_dir)
    if clean_output and outdir.exists():
        import shutil; shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results_list: List[pd.DataFrame] = []
    for s in tqdm(schemes, desc="Schemes", disable=not progress, leave=True):
        df_s = run_scheme_parallel(
            base=base, scheme=s,
            pools=pools, R_imp=R_imp, R_model=R_model, R_eval=R_eval,
            fixed_seeds=fixed_seeds,
            n_jobs=n_jobs, progress=progress,
            master_seed=master_seed, vary_all_max_combos=vary_all_max_combos,
            # pass knobs downstream
            M_imputations=M_imputations,
            imputer_max_iter=imputer_max_iter, imputer_tol=imputer_tol,
            test_size=test_size, cv_folds=cv_folds,
            n_estimators=n_estimators, subforest_size=subforest_size,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth, max_features=max_features,
        )
        results_list.append(df_s)

    results = pd.concat(results_list, ignore_index=True)
    assert set(results["scheme"].unique()).issubset(ALLOWED_SCHEMES)

    summary_df = summarize(results)

    # Save CSVs and metadata
    config_meta = dict(
        fixed_seeds=dict(imp=int(fixed_seeds[0]), model=int(fixed_seeds[1]), eval=int(fixed_seeds[2])),
        pool_sizes=dict(R_imp=R_imp, R_model=R_model, R_eval=R_eval),
        use_seed_file=use_seed_file,
        vary_all_max_combos=vary_all_max_combos,
        mi=dict(M_imputations=M_imputations, imputer_max_iter=imputer_max_iter, imputer_tol=imputer_tol),
        cf=dict(n_estimators=n_estimators, subforest_size=subforest_size,
                min_samples_leaf=min_samples_leaf, max_depth=max_depth, max_features=max_features,
                cv_folds=cv_folds),
        split=dict(test_size=test_size),
        dgp=dict(N=N, d=d, miss_rate=miss_rate, tau=tau, alloc_seed=alloc_seed),
    )
    save_outputs_csv_only(results, summary_df, outdir, config_meta)

    # Console summary
    pd.set_option("display.width", 160)
    print("\nSummary by scheme (ate_min/ate_max + ALL_EXCL_vary_all):")
    print(summary_df.to_string(index=False))
    print(f"\nSaved CSV outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
