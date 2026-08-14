"""
Compound Seed Variability Analysis for ML Pipeline

Author: Mark Verhagen

Description: Seed variability analysis of a simple ML pipeline with:
1. Imputation (stochastic random sampling from observed values)
2. Train/test split (stochastic)
3. Model training (deterministic LinearRegression - seed kept for API consistency)

Seeds are loaded from assets/seed_list.txt and results are saved to CSV.
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed


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


def generate_data(n_samples=2000, n_features=20, noise_std=1.0,
                  missing_rate=0.3, base_seed=42):
    """
    Generate a synthetic regression dataset with missing values.
    """
    rng = np.random.default_rng(base_seed)

    # Features
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))

    # True coefficients
    beta = rng.normal(loc=0.0, scale=1.0, size=n_features)

    # Targets
    y = X @ beta + rng.normal(loc=0.0, scale=noise_std, size=n_samples)

    # Introduce missing values at random (MCAR)
    mask = rng.uniform(size=X.shape) < missing_rate
    X_missing = X.copy()
    X_missing[mask] = np.nan

    return X_missing, y


def random_impute(X_with_nan, seed_impute):
    """
    Simple stochastic imputation:
    For each feature, replace NaNs by random samples (with replacement)
    from the observed (non-NaN) values in that feature.
    """
    rng = np.random.default_rng(seed_impute)
    X = X_with_nan.copy()
    n_samples, n_features = X.shape

    for j in range(n_features):
        col = X[:, j]
        missing_mask = np.isnan(col)
        if not np.any(missing_mask):
            continue  # no missing in this column

        observed = col[~missing_mask]
        if observed.size == 0:
            fill_values = np.zeros(np.sum(missing_mask))
        else:
            fill_values = rng.choice(observed, size=np.sum(missing_mask), replace=True)

        col[missing_mask] = fill_values
        X[:, j] = col

    return X


def process_single_run(seed_impute, seed_split, seed_train, X_missing, y):
    """
    Executes a single run of the pipeline: Imputation -> Split -> Model.
    Returns dict with seeds and MSE.
    """
    # 1. Impute
    X_imp = random_impute(X_missing, seed_impute)

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imp,
        y,
        test_size=0.2,
        random_state=seed_split
    )

    # 3. Train model
    model = MLPRegressor(
        hidden_layer_sizes=(50,),
        max_iter=300,
        random_state=seed_train
    )
    model.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return {
        'impute_seed': seed_impute,
        'split_seed': seed_split,
        'train_seed': seed_train,
        'mse': mse
    }


def run_pipeline(n_impute_seeds=100, n_split_seeds=100, n_train_seeds=1,
                 out_path="compound_sim_results"):
    """
    Run the compound seed variability analysis.
    """
    # Load seeds from file
    n_total_seeds = n_impute_seeds + n_split_seeds + n_train_seeds
    all_seeds = load_seeds(n_total_seeds)
    
    seeds_impute = all_seeds[:n_impute_seeds]
    seeds_split = all_seeds[n_impute_seeds : n_impute_seeds + n_split_seeds]
    seeds_train = all_seeds[n_impute_seeds + n_split_seeds :]
    
    
    # Generate base dataset (fixed)
    X_missing, y = generate_data()
    
    # Generate task list
    tasks = []
    for s_imp in seeds_impute:
        for s_split in seeds_split:
            for s_train in seeds_train:
                tasks.append((s_imp, s_split, s_train))
    
    
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_single_run)(
            seed_impute, seed_split, seed_train, X_missing, y
        ) for seed_impute, seed_split, seed_train in tasks
    )
    
    print("Training complete.")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = f"{out_path}_{n_impute_seeds}_{n_split_seeds}_{n_train_seeds}.csv"
    results_df.to_csv(output_file, index=False)
    
    return results_df

if __name__ == "__main__":
    run_pipeline(n_impute_seeds=1000, n_split_seeds=1000, n_train_seeds=1000,
                 out_path="../data/compound/compound_sim_stability_all_nn")
