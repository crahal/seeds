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
from sklearn.impute import IterativeImputer
from econml.dml import CausalForestDML
from joblib import Parallel, delayed

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

def process_single_run(
    seed_imp, seed_samp, seed_model,
    X_train_raw_full, T_train_full, Y_train_full, X_holdout_raw, params):
    
    """
    Executes a single run of the pipeline: Imputation -> Sampling -> Model.
    """
    
    B, min_leaf = params
    
    # 1. Impute missing data using the IterativeImputer with sample_posterior = True
    imputer = IterativeImputer(max_iter=10, random_state=seed_imp, sample_posterior=True)
    X_train_imputed_np = imputer.fit_transform(X_train_raw_full)
    X_train_imputed_full = pd.DataFrame(X_train_imputed_np, columns=X_train_raw_full.columns, index=X_train_raw_full.index)
    
    # Transform holdout set
    X_holdout_imputed_np = imputer.transform(X_holdout_raw)
    X_holdout_imputed = pd.DataFrame(X_holdout_imputed_np, columns=X_train_raw_full.columns, index=X_holdout_raw.index)
    
    # 2. Resample the training set to simulate standard train / test splitting
    n_train = len(X_train_raw_full)
    rng_samp = np.random.default_rng(seed_samp)
    indices = rng_samp.choice(n_train, size=n_train, replace=True)
    
    X_train_resampled = X_train_imputed_full.iloc[indices]
    T_train_resampled = T_train_full.iloc[indices]
    Y_train_resampled = Y_train_full.iloc[indices]
    
    # 3. Model
    cf = CausalForestDML(
        n_estimators=B,
        min_samples_leaf=min_leaf,
        honest=True,
        discrete_treatment=True,
        random_state=seed_model,
        verbose=0,
        n_jobs=1 
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

def load_seeds(n_needed, filepath="./assets/seed_list.txt"):
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
    
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i >= n_needed:
                    break
                line = line.strip()
                if line:
                    seeds.append(int(line))
    except Exception as e:
        print(f"Error reading seed file: {e}")
        
    return seeds

def run_pipeline(n_impute_seeds=1, n_sample_seeds=1, n_model_seeds=1,
                 out_path="brand_ate_stability_nested.csv", prop_holdout=0.2):
    ## Set parameters
    B = 2000 
    min_leaf = 20 
    params = (B, min_leaf)
    
    data_path = "./data/hte/brand_et_al_with_missing_sample.csv"
    print(f"Loading data from {data_path}...")
    
    # Handle path resolution for running from src or from root
    if not os.path.exists(data_path):
        # Try going up one level
        potential_path = os.path.join("..", data_path)
        if os.path.exists(potential_path):
            data_path = potential_path
        else:
             pass

    try:
        X_raw, T, Y = load_data_raw(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return

    print(f"Data loaded: N={len(Y)}, Features={X_raw.shape[1]}")
    
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
    print(f"Loading {n_total_seeds} seeds from assets...")
    all_seeds = load_seeds(n_total_seeds)
    
    seeds_imp = all_seeds[:n_impute_seeds]
    seeds_samp = all_seeds[n_impute_seeds : n_impute_seeds + n_sample_seeds]
    seeds_model = all_seeds[n_impute_seeds + n_sample_seeds :]
    
    # Generate task list
    tasks = []
    for s_imp in seeds_imp:
        for s_samp in seeds_samp:
            for s_model in seeds_model:
                tasks.append((s_imp, s_samp, s_model))
    
    print(f"Running {len(tasks)} tasks in parallel...")
    
    results = Parallel(n_jobs=-1)(
        delayed(process_single_run)(
            seed_imp, seed_samp, seed_model, 
            X_train_raw_full, T_train_full, Y_train_full, X_holdout_raw, params
        ) for seed_imp, seed_samp, seed_model in tasks
    )
    
    print("Training complete.")
    
    # Save results
    results_df = pd.DataFrame(results)
    print(f"Saving results to {f'{out_path}_{len(results)}_{prop_holdout}.csv'}")
    results_df.to_csv(f"{out_path}_{len(results)}_{prop_holdout}.csv", index=False)
    
    return results_df

if __name__ == "__main__":
    # Default configuration or parse args
    run_pipeline(n_impute_seeds=1000, n_sample_seeds=1, n_model_seeds=1,
                 out_path="brand_ate_stability_impute", prop_holdout=0.3)
    run_pipeline(n_impute_seeds=1, n_sample_seeds=1000, n_model_seeds=1,
                 out_path="brand_ate_stability_sample", prop_holdout=0.3)
    run_pipeline(n_impute_seeds=1, n_sample_seeds=1, n_model_seeds=1000,
                 out_path="brand_ate_stability_model", prop_holdout=0.3)
    run_pipeline(n_impute_seeds=10, n_sample_seeds=10, n_model_seeds=10,
                 out_path="brand_ate_stability_all", prop_holdout=0.3)
    