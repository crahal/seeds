"""
Reproduction of Uncovering Sociological Effect Heterogeneity (Brand et al.)

Reference: Athey & Wager (2015), Wager & Athey (2018).

Author: Mark Verhagen

Date: 2025-11-25

Description: Seed variability analysis of a standard causal forest pipeline.
Dataset is based on the paper by Brand et al. (2021) but includes artificial
missingness (see data preprocessing script in ./data/hte/)

The last `prop_holdout` rows are used as a stable holdout set to calculate the
ATE for. Seed variability comes from:

1. Imputing the missing data using the InterativeImputer with sample_posterior=True
2. Sampling the training sample with replacement (to simluate standard train / test splitting)
3. Model estimation of the causal forest
"""

import numpy as np
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from econml.dml import CausalForestDML
from joblib import Parallel, delayed
from tqdm import tqdm


def load_data_raw(filepath):
    """
    Loads the Brand et al. sample data with constructed missingness.
    """
    df = pd.read_csv(filepath)
    
    # Linear covariates used in the original paper
    covariates = [
        "male","black","hisp","i_daded","i_momed","i_parinc","i_daduwhcol",
        "i_intact","i_sibsz","i_rural","i_south","i_abil","i_hsprog",
        "i_eduexp","i_eduasp","i_freduasp","i_rotter","i_delinq",
        "i_schdisadv","i_mar18","i_parent18","good"
    ]
    
    # Treatment indicator used to study ATE and CATE
    treatment_indicator = "compcoll25"
    
    # Outcome variable of interest
    outcome_variable = "lowwaprop"
    
    # No missing values in treatment and outcome
    df = df.dropna(subset=[treatment_indicator, outcome_variable])
    
    # Subset data
    X_raw = df[covariates]
    T = df[treatment_indicator]
    Y = df[outcome_variable]
    
    return X_raw, T, Y

def impute_data(seed_imp, X_train_raw_full, X_holdout_raw):
    """
    Pre-compute imputation for a given seed. This is separated out to avoid
    redundant imputation when the same impute_seed is used with different
    sample/model seeds.
    """
    imputer = IterativeImputer(max_iter=10, random_state=seed_imp, sample_posterior=True)
    X_train_imputed_np = imputer.fit_transform(X_train_raw_full)
    X_train_imputed = pd.DataFrame(X_train_imputed_np, columns=X_train_raw_full.columns, index=X_train_raw_full.index)
    
    X_holdout_imputed_np = imputer.transform(X_holdout_raw)
    X_holdout_imputed = pd.DataFrame(X_holdout_imputed_np, columns=X_train_raw_full.columns, index=X_holdout_raw.index)
    
    return seed_imp, X_train_imputed, X_holdout_imputed


def process_single_run(
    seed_imp, seed_samp, seed_model,
    X_train_imputed, T_train_full, Y_train_full, X_holdout_imputed, params):
    """
    Executes a single run of the pipeline: Sampling -> Model.
    Imputation is now pre-computed and passed in.
    """
    
    B, min_leaf = params
    
    # 1. Resample the training set to simulate standard train / test splitting
    n_train = len(X_train_imputed)
    rng_samp = np.random.default_rng(seed_samp)
    indices = rng_samp.choice(n_train, size=n_train, replace=True)
    
    X_train_resampled = X_train_imputed.iloc[indices]
    T_train_resampled = T_train_full.iloc[indices]
    Y_train_resampled = Y_train_full.iloc[indices]
    
    # 2. Model
    treatment_model = LogisticRegression(
        max_iter=2000, solver="lbfgs", n_jobs=1
    )  # higher max_iter to avoid convergence warnings
    cf = CausalForestDML(
        n_estimators=B,
        min_samples_leaf=min_leaf,
        honest=True,
        discrete_treatment=True,
        random_state=seed_model,
        verbose=0,
        n_jobs=1,
        model_t=treatment_model,
    )
    
    cf.fit(Y_train_resampled, T_train_resampled, X=X_train_resampled)
    
    # Predict CATE on holdout set and calculate average treatment effect
    tau_hat = cf.effect(X_holdout_imputed)
    ate = np.mean(tau_hat)
    
    return {
        'impute_seed': seed_imp,
        'sample_seed': seed_samp,
        'model_seed': seed_model,
        'ate': ate
    }

def load_seeds(n_needed, filepath="../assets/seed_list.txt"):
    """
    Loads the first n_needed seeds from the seed list file.
    """
    seeds = []
    
    # Handle path resolution for running from src or from root
    if not os.path.exists(filepath):
        # Try going up one level
        potential_path = os.path.join("..", filepath)
        if os.path.exists(potential_path):
            filepath = potential_path
    
    if not os.path.exists(filepath):
        print(f"Warning: Seed file not found at {filepath} from {os.abspath('')}. Setting seeds to range({n_needed})")
        return list(range(n_needed))
    
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_needed:
                break
            line = line.strip()
            if line:
                seeds.append(int(line))
        
    return seeds

def run_pipeline(n_impute_seeds=1, n_sample_seeds=1, n_model_seeds=1,
                 out_path="brand_ate_stability_nested.csv", prop_holdout=0.2,
                 n_jobs=-1):
    """
    Run the full pipeline
    
    Key optimization: Imputations are pre-computed once per unique impute_seed,
    avoiding redundant computation when the same impute_seed is used with
    different sample/model seeds.
    """
    start_time = time.time()
    
    ## Set parameters
    B = 500
    min_leaf = 20 
    params = (B, min_leaf)
    
    n_total_tasks = n_impute_seeds * n_sample_seeds * n_model_seeds
    n_model_tasks = n_sample_seeds * n_model_seeds  # Tasks per imputation
    
    data_path = "../data/compound/brand_et_al_with_missing_sample.csv"
    
    # Handle path resolution for running from src or from root
    if not os.path.exists(data_path):
        potential_path = os.path.join(data_path)
        if os.path.exists(potential_path):
            data_path = potential_path

    X_raw, T, Y = load_data_raw(data_path)

    
    # 1. Split Holdout (Last `prop_holdout`%) - Keep fixed
    n_total = len(X_raw)
    n_holdout = int(n_total * prop_holdout)
    n_train = n_total - n_holdout
    
    X_train_raw_full = X_raw.iloc[:n_train]
    T_train_full = T.iloc[:n_train]
    Y_train_full = Y.iloc[:n_train]
    X_holdout_raw = X_raw.iloc[n_train:]
    
    # Load Seeds from file
    n_total_seeds = n_impute_seeds + n_sample_seeds + n_model_seeds
    all_seeds = load_seeds(n_total_seeds)
    
    seeds_imp = all_seeds[:n_impute_seeds]
    seeds_samp = all_seeds[n_impute_seeds : n_impute_seeds + n_sample_seeds]
    seeds_model = all_seeds[n_impute_seeds + n_sample_seeds :]
    
    # =========================================================================
    # PHASE 1: Pre-compute all imputations
    # =========================================================================
    impute_start = time.time()
    
    imputed_datasets = Parallel(n_jobs=n_jobs)(
        delayed(impute_data)(seed_imp, X_train_raw_full, X_holdout_raw)
        for seed_imp in tqdm(seeds_imp, desc="Imputing", unit="seed")
    )
    
    # Convert to dict for fast lookup
    imputed_cache = {seed: (X_train, X_holdout) for seed, X_train, X_holdout in imputed_datasets}
    
    impute_time = time.time() - impute_start
    
    # =========================================================================
    # PHASE 2: Run model fitting with progress tracking
    # =========================================================================
    model_start = time.time()
    all_results = []
    
    # Process each imputation group with progress tracking
    for imp_idx, seed_imp in enumerate(seeds_imp):
        X_train_imputed, X_holdout_imputed = imputed_cache[seed_imp]
        
        # Generate tasks for this imputation
        tasks = [
            (seed_imp, s_samp, s_model)
            for s_samp in seeds_samp
            for s_model in seeds_model
        ]
        
        # Estimate time remaining
        if imp_idx > 0:
            elapsed = time.time() - model_start
            rate = (imp_idx * n_model_tasks) / elapsed
            remaining_tasks = (n_impute_seeds - imp_idx) * n_model_tasks
            eta_seconds = remaining_tasks / rate if rate > 0 else 0
            eta = datetime.now() + timedelta(seconds=eta_seconds)
            eta_str = eta.strftime("%H:%M:%S")
        
        # Run tasks for this imputation in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_run)(
                seed_imp, seed_samp, seed_model,
                X_train_imputed, T_train_full, Y_train_full, X_holdout_imputed, params
            ) for seed_imp, seed_samp, seed_model in tasks
        )
        
        all_results.extend(results)
    
    model_time = time.time() - model_start
    total_time = time.time() - start_time
    
    # =========================================================================
    # Summary and save
    # =========================================================================
    
    # Save results
    results_df = pd.DataFrame(all_results)
    output_file = f"{out_path}_{n_impute_seeds}_{n_sample_seeds}_{n_model_seeds}.csv"
    results_df.to_csv(output_file, index=False) 
    
    return results_df

if __name__ == "__main__":
    run_pipeline(n_impute_seeds=1, n_sample_seeds=1, n_model_seeds=1,
                 out_path="../data/compound/brand_ate_stability_all", prop_holdout=0.3)
    
