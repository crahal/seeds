#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2-fold OOF CV for RF + Logistic + shallow NN (MLP) on ProPublica COMPAS data.

- Non-stratified KFold (N_SPLITS=2), identical seed used for fold split and all models.
- Parallelized across seeds with joblib (process-based), RF also uses n_jobs=-1 internally.
- Saves DataFrames.
"""

import os
import io
import argparse
import urllib.request
from typing import List

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from joblib import Parallel, delayed

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score


# ----------------------------- Configuration defaults -----------------------------
DATA_URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
N_SPLITS_DEFAULT = 2
THRESH_DEFAULT   = 0.5
N_SEEDS_DEFAULT  = 10000


# ----------------------------- Utilities -----------------------------
def load_and_preprocess(url: str):
    with urllib.request.urlopen(url) as resp:
        df = pd.read_csv(io.BytesIO(resp.read()))
    df = df[
        (df["days_b_screening_arrest"] <= 30) &
        (df["days_b_screening_arrest"] >= -30) &
        (df["is_recid"] != -1) &
        (df["c_charge_degree"] != "O") &
        (df["score_text"] != "N/A")
    ].copy()
    assert "id" in df.columns, "Expected a column named 'id' to serve as UID."
    if df["id"].duplicated().any():
        df["id"] = df["id"].astype(str) + "_" + df.groupby("id").cumcount().astype(str)
    X = df[["age", "priors_count"]].copy()
    y = df["two_year_recid"].astype(int).to_numpy()
    uid = df["id"].astype(str).to_numpy()
    compas_decile = df["decile_score"].to_numpy()
    compas_hat = (compas_decile >= 5).astype(int)
    return df, X, y, uid, compas_decile, compas_hat


def per_uid_stats(oof: np.ndarray, thr: float):
    mu   = np.nanmean(oof, axis=1)
    sd   = np.nanstd(oof, axis=1, ddof=1)
    q10  = np.nanpercentile(oof, 10, axis=1)
    q50  = np.nanpercentile(oof, 50, axis=1)
    q90  = np.nanpercentile(oof, 90, axis=1)
    bin_ = (oof >= thr).astype(int)
    ones = bin_.sum(axis=1)
    flips = np.minimum(ones, oof.shape[1] - ones)
    fliprate = flips / oof.shape[1]
    return mu, sd, q10, q50, q90, flips, fliprate


def mean_std(a: List[float]):
    arr = np.asarray(a, float)
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1))


# ----------------------------- Core OOF runner (single seed) -----------------------------
def run_oof_one_seed(X: pd.DataFrame, y: np.ndarray, seed: int, n_splits: int):
    """
    Non-stratified 2-fold OOF probabilities for RF, LR, and shallow NN (MLP) with fixed random_state=seed.
    For each i, \hat{p}_i is computed from a model trained on a split that excludes i.
    """
    n = len(y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_rf = np.full(n, np.nan, dtype=float)
    oof_lr = np.full(n, np.nan, dtype=float)
    oof_nn = np.full(n, np.nan, dtype=float)

    for tr, va in kf.split(X, y):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr       = y[tr]

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=5, random_state=seed, n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        oof_rf[va] = rf.predict_proba(X_va)[:, 1]

        # Logistic Regression
        lr = LogisticRegression(
            solver="saga", penalty="l2", max_iter=2000, random_state=seed
        )
        lr.fit(X_tr, y_tr)
        oof_lr[va] = lr.predict_proba(X_va)[:, 1]

        # Shallow NN (MLP) with standardization
        mlp = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            MLPClassifier(
                hidden_layer_sizes=(16,),
                activation="relu",
                solver="lbfgs",
                alpha=1e-4,
                max_iter=2000,
                random_state=seed
            )
        )
        mlp.fit(X_tr, y_tr)
        oof_nn[va] = mlp.predict_proba(X_va)[:, 1]

    return oof_rf, oof_lr, oof_nn


# ----------------------------- Main -----------------------------
def main():
    parser = argparse.ArgumentParser(description="2-fold OOF CV for RF, LR, and shallow NN on COMPAS.")
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS_DEFAULT, help="Number of seeds (default: 100).")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory to save CSVs and plots.")
    parser.add_argument("--threshold", type=float, default=THRESH_DEFAULT, help="Classification threshold (default: 0.5).")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS_DEFAULT, help="Number of folds (default: 2).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Load + preprocess
    df, X, y, uid, compas_decile, compas_hat = load_and_preprocess(DATA_URL)

    # Seeds
    seeds = list(range(args.n_seeds))

    # Parallel OOF across seeds
    print(f"[INFO] Running {args.n_splits}-fold OOF across {args.n_seeds} seeds (parallelized)...")
    results = Parallel(n_jobs=-1, backend="loky", verbose=0)(
        delayed(run_oof_one_seed)(X, y, seed=s, n_splits=args.n_splits) for s in tqdm(seeds)
    )

    # Stack into (n, S)
    oof_rf_all = np.column_stack([res[0] for res in results])
    oof_lr_all = np.column_stack([res[1] for res in results])
    oof_nn_all = np.column_stack([res[2] for res in results])

    # Seed-wise metrics
    auc_rf_seeds = [roc_auc_score(y, oof_rf_all[:, j]) for j in range(args.n_seeds)]
    acc_rf_seeds = [accuracy_score(y, (oof_rf_all[:, j] >= args.threshold).astype(int)) for j in range(args.n_seeds)]
    auc_lr_seeds = [roc_auc_score(y, oof_lr_all[:, j]) for j in range(args.n_seeds)]
    acc_lr_seeds = [accuracy_score(y, (oof_lr_all[:, j] >= args.threshold).astype(int)) for j in range(args.n_seeds)]
    auc_nn_seeds = [roc_auc_score(y, oof_nn_all[:, j]) for j in range(args.n_seeds)]
    acc_nn_seeds = [accuracy_score(y, (oof_nn_all[:, j] >= args.threshold).astype(int)) for j in range(args.n_seeds)]

    # UID-indexed wide output
    cols_rf = {f"y_hat_rf_seed{seeds[j]:03d}": oof_rf_all[:, j] for j in range(args.n_seeds)}
    cols_lr = {f"y_hat_lr_seed{seeds[j]:03d}": oof_lr_all[:, j] for j in range(args.n_seeds)}
    cols_nn = {f"y_hat_nn_seed{seeds[j]:03d}": oof_nn_all[:, j] for j in range(args.n_seeds)}
    pred_df = pd.DataFrame(
        {"y": y, "compas_decile": compas_decile, "compas_hat": compas_hat, **cols_rf, **cols_lr, **cols_nn},
        index=uid,
    )
    pred_df.index.name = "UID"
    pred_path = os.path.join(data_dir, f"uid_oof_predictions_{args.n_seeds}seeds_rf_lr_nn_compas_{args.n_splits}folds.csv")
    pred_df.to_csv(pred_path)
    print(f"[INFO] Saved predictions to {pred_path}")

    # Global summaries
    def _ms(x): return mean_std(x)
    auc_rf_mean, auc_rf_std = _ms(auc_rf_seeds)
    acc_rf_mean, acc_rf_std = _ms(acc_rf_seeds)
    auc_lr_mean, auc_lr_std = _ms(auc_lr_seeds)
    acc_lr_mean, acc_lr_std = _ms(acc_lr_seeds)
    auc_nn_mean, auc_nn_std = _ms(auc_nn_seeds)
    acc_nn_mean, acc_nn_std = _ms(acc_nn_seeds)

    auc_compas = roc_auc_score(y, compas_decile)
    acc_compas = accuracy_score(y, compas_hat)

    print("=== Across seeds (2-fold OOF) ===")
    print(f"RF  (age+priors):  AUC = {auc_rf_mean:.3f} ± {auc_rf_std:.3f},  ACC({args.threshold:.2f}) = {acc_rf_mean:.3f} ± {acc_rf_std:.3f}")
    print(f"LR  (age+priors):  AUC = {auc_lr_mean:.3f} ± {auc_lr_std:.3f},  ACC({args.threshold:.2f}) = {acc_lr_mean:.3f} ± {acc_lr_std:.3f}")
    print(f"NN  (MLP, 1×16):   AUC = {auc_nn_mean:.3f} ± {auc_nn_std:.3f},  ACC({args.threshold:.2f}) = {acc_nn_mean:.3f} ± {acc_nn_std:.3f}")
    print("\n=== COMPAS (seed-invariant) ===")
    print(f"AUC (decile 1..10): {auc_compas:.3f}")
    print(f"ACC (decile>=5):     {acc_compas:.3f}")

    # Per-UID instability summaries
    rf_mu, rf_sd, *_ , rf_fliprate = per_uid_stats(oof_rf_all, thr=args.threshold)
    lr_mu, lr_sd, *_ , lr_fliprate = per_uid_stats(oof_lr_all, thr=args.threshold)
    nn_mu, nn_sd, *_ , nn_fliprate = per_uid_stats(oof_nn_all, thr=args.threshold)

    age   = X["age"].to_numpy()
    priors= X["priors_count"].to_numpy()

    summary_df = pd.DataFrame({
        "y": y,
        # RF
        "rf_mu": rf_mu, "rf_sd": rf_sd, "rf_fliprate": rf_fliprate,
        # LR
        "lr_mu": lr_mu, "lr_sd": lr_sd, "lr_fliprate": lr_fliprate,
        # NN
        "nn_mu": nn_mu, "nn_sd": nn_sd, "nn_fliprate": nn_fliprate,
        # Features / COMPAS
        "age": age, "priors": priors, "compas_decile": compas_decile
    }, index=uid)
    summary_df.index.name = "UID"
    summary_path = os.path.join(data_dir, f"uid_summary_instability_{args.n_seeds}seeds_{args.n_splits}folds_rf_lr_nn.csv")
    summary_df.to_csv(summary_path)
    print(f"[INFO] Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
