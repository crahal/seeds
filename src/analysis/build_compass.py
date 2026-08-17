#!/usr/bin/env python3
"""Crossed S5 diagnostics and visualization runs for COMPAS recidivism.

The diagnostic experiment crosses shared nonparametric bootstrap datasets
with joint seed vectors.  One scalar seed controls the shuffled two-fold split
and every model random state, matching Supplementary Section S3.8.  A separate
1,000-seed sweep on the observed dataset supplies per-person OOF probabilities
for the figure and is never used to estimate sampling uncertainty, r, or rho.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import time
import urllib.request
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed, parallel_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .reporting_utils import (
        bootstrap_index_block,
        configure_run_logger,
        crossed_s5_diagnostics,
        format_observed_seed_report,
        format_s5_report,
        load_seed_list,
        observed_seed_summary,
        resolve_batch_size,
        resolve_worker_count,
        validate_s5_log,
    )
except ImportError:  # pragma: no cover - direct script execution
    from reporting_utils import (
        bootstrap_index_block,
        configure_run_logger,
        crossed_s5_diagnostics,
        format_observed_seed_report,
        format_s5_report,
        load_seed_list,
        observed_seed_summary,
        resolve_batch_size,
        resolve_worker_count,
        validate_s5_log,
    )


DATA_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/master/"
    "compas-scores-two-years.csv"
)
MODEL_KEYS = ("lr", "rf", "nn")
MODEL_LABELS = {
    "lr": "Logistic regression",
    "rf": "Random forest",
    "nn": "Shallow neural network",
}
METRIC_KEYS = ("auc", "accuracy")
METRIC_LABELS = {"auc": "AUC", "accuracy": "Accuracy"}
SCORE_KEYS = tuple(
    f"{model}_{metric}" for model in MODEL_KEYS for metric in METRIC_KEYS
)


@dataclass(frozen=True)
class Settings:
    n_bootstrap_resamples: int = 100
    n_seed_vectors: int = 100
    n_visualization_runs: int = 1_000
    n_splits: int = 2
    threshold: float = 0.5
    rf_n_estimators: int = 200
    rf_max_depth: int = 5
    model_max_iter: int = 2_000
    mlp_hidden_units: int = 16
    n_jobs: int = -1
    bootstrap_batch_size: int = 0
    observed_batch_size: int = 200

    def validate(self) -> None:
        if self.n_bootstrap_resamples < 2 or self.n_seed_vectors < 2:
            raise ValueError("S5 diagnostics require at least two bootstraps and seeds")
        if self.n_visualization_runs < 2:
            raise ValueError("n_visualization_runs must be at least two")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least two")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must lie strictly between zero and one")
        if self.rf_n_estimators < 1 or self.rf_max_depth < 1:
            raise ValueError("random-forest controls must be positive")
        if self.model_max_iter < 1 or self.mlp_hidden_units < 1:
            raise ValueError("model iteration and hidden-unit controls must be positive")
        if self.bootstrap_batch_size < 0 or self.observed_batch_size < 0:
            raise ValueError("checkpoint batch sizes cannot be negative")
        resolve_worker_count(self.n_jobs)


@dataclass(frozen=True)
class SeedPlan:
    joint: np.ndarray
    bootstrap: np.ndarray


def make_seed_plan(seed_list: Sequence[int], settings: Settings) -> SeedPlan:
    """Allocate one joint fold/model seed block and a disjoint bootstrap block."""

    n_joint = max(settings.n_seed_vectors, settings.n_visualization_runs)
    stop = n_joint + settings.n_bootstrap_resamples
    if len(seed_list) < stop:
        raise ValueError(f"seed list has {len(seed_list)} values but {stop} are required")
    selected = np.asarray(seed_list[:stop], dtype=np.uint64)
    if len(np.unique(selected)) != len(selected):
        raise ValueError("selected algorithmic and bootstrap seeds must be unique")
    joint = selected[:n_joint].copy()
    bootstrap = selected[n_joint:stop].copy()
    if np.intersect1d(joint, bootstrap).size:
        raise ValueError("algorithmic and bootstrap seed blocks must be disjoint")
    return SeedPlan(joint=joint, bootstrap=bootstrap)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _ensure_input(path: Path, url: str) -> Path:
    """Cache the public ProPublica input without replacing a valid local copy."""

    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    try:
        urllib.request.urlretrieve(url, temporary)
        pd.read_csv(temporary, nrows=2)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def load_and_preprocess(source: str | Path):
    """Load and reproduce the ProPublica two-year COMPAS cohort filter."""

    df = pd.read_csv(source)
    df = df[
        df["days_b_screening_arrest"].between(-30, 30)
        & df["is_recid"].ne(-1)
        & df["c_charge_degree"].ne("O")
        & df["score_text"].ne("N/A")
    ].copy()
    if "id" not in df:
        raise ValueError("COMPAS input must contain an id column")
    if df["id"].duplicated().any():
        df["id"] = (
            df["id"].astype(str)
            + "_"
            + df.groupby("id").cumcount().astype(str)
        )
    X = df[["age", "priors_count"]].astype(float).copy()
    y = df["two_year_recid"].astype(int).to_numpy()
    uid = df["id"].astype(str).to_numpy()
    compas_decile = df["decile_score"].to_numpy(dtype=float)
    compas_hat = (compas_decile >= 5).astype(int)
    if len(df) < 4 or set(np.unique(y)) != {0, 1}:
        raise ValueError("filtered COMPAS cohort must contain both outcome classes")
    if not np.isfinite(X.to_numpy()).all() or not np.isfinite(compas_decile).all():
        raise ValueError("COMPAS features and scores must be finite")
    return df, X, y, uid, compas_decile, compas_hat


def _validate_predictions(values: np.ndarray, *, model: str) -> np.ndarray:
    predictions = np.asarray(values, dtype=float)
    if predictions.ndim != 1 or not np.isfinite(predictions).all():
        raise RuntimeError(f"{model} produced incomplete OOF probabilities")
    if np.any((predictions < 0) | (predictions > 1)):
        raise RuntimeError(f"{model} produced probabilities outside [0, 1]")
    return predictions


def run_oof_one_seed(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    seed: int,
    n_splits: int,
    *,
    rf_n_estimators: int = 200,
    rf_max_depth: int = 5,
    model_max_iter: int = 2_000,
    mlp_hidden_units: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return RF, LR, and MLP OOF probabilities for one joint seed vector."""

    features = np.asarray(X, dtype=float)
    target = np.asarray(y, dtype=int)
    if features.ndim != 2 or len(features) != len(target):
        raise ValueError("X and y must be aligned two-dimensional/one-dimensional arrays")
    if n_splits > len(target):
        raise ValueError("n_splits cannot exceed the number of observations")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    predictions = {
        model: np.full(len(target), np.nan, dtype=float) for model in MODEL_KEYS
    }
    for training, validation in splitter.split(features, target):
        y_training = target[training]
        if len(np.unique(y_training)) != 2:
            raise RuntimeError("a CV training fold contains only one outcome class")
        X_training, X_validation = features[training], features[validation]

        rf = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=int(seed),
            n_jobs=1,
        )
        rf.fit(X_training, y_training)
        predictions["rf"][validation] = rf.predict_proba(X_validation)[:, 1]

        lr = LogisticRegression(
            solver="saga",
            penalty="l2",
            max_iter=model_max_iter,
            random_state=int(seed),
        )
        lr.fit(X_training, y_training)
        predictions["lr"][validation] = lr.predict_proba(X_validation)[:, 1]

        mlp = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            MLPClassifier(
                hidden_layer_sizes=(mlp_hidden_units,),
                activation="relu",
                solver="lbfgs",
                alpha=1e-4,
                max_iter=model_max_iter,
                random_state=int(seed),
            ),
        )
        mlp.fit(X_training, y_training)
        predictions["nn"][validation] = mlp.predict_proba(X_validation)[:, 1]

    return tuple(
        _validate_predictions(predictions[model], model=model)
        for model in ("rf", "lr", "nn")
    )


def _run_oof_audited(
    X: np.ndarray, y: np.ndarray, seed: int, settings: Settings
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rf, lr, nn = run_oof_one_seed(
            X,
            y,
            int(seed),
            settings.n_splits,
            rf_n_estimators=settings.rf_n_estimators,
            rf_max_depth=settings.rf_max_depth,
            model_max_iter=settings.model_max_iter,
            mlp_hidden_units=settings.mlp_hidden_units,
        )
    issue_counts = {
        "convergence_warnings": sum(
            issubclass(item.category, ConvergenceWarning) for item in caught
        ),
        "other_warnings": sum(
            not issubclass(item.category, ConvergenceWarning) for item in caught
        ),
    }
    return {"rf": rf, "lr": lr, "nn": nn}, issue_counts


def _scores_from_predictions(
    y: np.ndarray, predictions: Mapping[str, np.ndarray], threshold: float
) -> dict[str, float]:
    if len(np.unique(y)) != 2:
        raise RuntimeError("AUC requires both outcome classes")
    scores: dict[str, float] = {}
    for model in MODEL_KEYS:
        probabilities = _validate_predictions(predictions[model], model=model)
        scores[f"{model}_auc"] = float(roc_auc_score(y, probabilities))
        scores[f"{model}_accuracy"] = float(
            accuracy_score(y, (probabilities >= threshold).astype(int))
        )
    if not np.isfinite(list(scores.values())).all():
        raise RuntimeError("a COMPAS score is non-finite")
    return scores


def _evaluate_bootstrap_row(
    bootstrap_number: int,
    indices: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    seeds: np.ndarray,
    settings: Settings,
) -> tuple[int, dict[str, np.ndarray], dict[str, int]]:
    sample = indices[bootstrap_number]
    X_bootstrap, y_bootstrap = X[sample], y[sample]
    row_scores = {key: np.empty(len(seeds), dtype=float) for key in SCORE_KEYS}
    issues: Counter[str] = Counter()
    for seed_number, seed in enumerate(seeds):
        try:
            predictions, cell_issues = _run_oof_audited(
                X_bootstrap, y_bootstrap, int(seed), settings
            )
            scores = _scores_from_predictions(
                y_bootstrap, predictions, settings.threshold
            )
        except Exception as exc:
            raise RuntimeError(
                "COMPAS crossed cell failed at bootstrap "
                f"{bootstrap_number}, seed vector {seed_number}, seed {int(seed)}"
            ) from exc
        for key, value in scores.items():
            row_scores[key][seed_number] = value
        issues.update(cell_issues)
    return bootstrap_number, row_scores, dict(issues)


def _score_path(output_dir: Path, key: str) -> Path:
    return output_dir / f"compass_crossed_{key}_scores.npy"


def _load_array(path: Path, shape: tuple[int, ...], *, allow_nan: bool) -> np.ndarray:
    values = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if values.shape != shape:
        raise ValueError(f"{path} has shape {values.shape}; expected {shape}")
    if np.isinf(values).any() or (not allow_nan and not np.isfinite(values).all()):
        raise ValueError(f"{path} contains invalid values")
    finite = values[np.isfinite(values)]
    if finite.size and np.any((finite < 0) | (finite > 1)):
        raise ValueError(f"{path} contains scores/probabilities outside [0, 1]")
    return values


def _load_issue_counts(path: Path, *, resume: bool) -> Counter[str]:
    if not resume or not path.exists():
        return Counter()
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return Counter({str(key): int(value) for key, value in payload.items()})


def _save_crossed_state(
    output_dir: Path,
    matrices: Mapping[str, np.ndarray],
    issue_counts: Mapping[str, int],
) -> None:
    for key, values in matrices.items():
        _atomic_save_array(_score_path(output_dir, key), values)
    _atomic_write_json(
        output_dir / "compass_crossed_issue_counts.json", issue_counts
    )


def run_crossed_grid(
    X: np.ndarray,
    y: np.ndarray,
    bootstrap_indices: np.ndarray,
    seed_plan: SeedPlan,
    settings: Settings,
    *,
    output_dir: Path,
    logger: Any,
    resume: bool,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    B, m = settings.n_bootstrap_resamples, settings.n_seed_vectors
    shape = (B, m)
    matrices: dict[str, np.ndarray] = {}
    for key in SCORE_KEYS:
        path = _score_path(output_dir, key)
        matrices[key] = (
            _load_array(path, shape, allow_nan=True)
            if resume and path.exists()
            else np.full(shape, np.nan, dtype=float)
        )

    complete = np.stack([np.isfinite(values) for values in matrices.values()])
    complete_by_row = complete.reshape(len(SCORE_KEYS), B, m)
    for row in range(B):
        row_state = complete_by_row[:, row, :]
        if row_state.any() and not row_state.all():
            raise ValueError(f"crossed checkpoint has partial bootstrap row {row}")
    pending = [row for row in range(B) if not complete_by_row[:, row, :].all()]
    issue_path = output_dir / "compass_crossed_issue_counts.json"
    issues = _load_issue_counts(issue_path, resume=resume)
    if not pending:
        return matrices, dict(issues)

    workers = resolve_worker_count(settings.n_jobs, n_tasks=len(pending))
    batch = resolve_batch_size(
        len(pending), batch_size=settings.bootstrap_batch_size, n_jobs=workers
    )
    seeds = seed_plan.joint[:m]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        for start in range(0, len(pending), batch):
            rows = pending[start : start + batch]
            results = Parallel(n_jobs=workers, batch_size=1)(
                delayed(_evaluate_bootstrap_row)(
                    row, bootstrap_indices, X, y, seeds, settings
                )
                for row in rows
            )
            for row, row_scores, row_issues in results:
                for key, values in row_scores.items():
                    matrices[key][row] = values
                issues.update(row_issues)
            _save_crossed_state(output_dir, matrices, issues)
            completed = B - sum(
                not np.isfinite(matrices[SCORE_KEYS[0]][row]).all()
                for row in range(B)
            )
            logger.info(
                "Crossed grid: %d/%d bootstrap rows (%d/%d cells) complete",
                completed,
                B,
                completed * m,
                B * m,
            )
    if any(not np.isfinite(values).all() for values in matrices.values()):
        raise RuntimeError("crossed COMPAS score matrices are incomplete")
    return matrices, dict(issues)


def _evaluate_observed_seed(
    seed_number: int,
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    settings: Settings,
) -> tuple[int, dict[str, np.ndarray], dict[str, int]]:
    try:
        predictions, issues = _run_oof_audited(X, y, int(seed), settings)
    except Exception as exc:
        raise RuntimeError(
            f"COMPAS observed-data seed vector {seed_number}, seed {int(seed)} failed"
        ) from exc
    return seed_number, predictions, issues


def _observed_path(output_dir: Path, model: str) -> Path:
    return output_dir / f"compass_observed_{model}_oof_probabilities.npy"


def _save_observed_state(
    output_dir: Path,
    predictions: Mapping[str, np.ndarray],
    issue_counts: Mapping[str, int],
) -> None:
    for model, values in predictions.items():
        _atomic_save_array(_observed_path(output_dir, model), values)
    _atomic_write_json(
        output_dir / "compass_observed_issue_counts.json", issue_counts
    )


def run_observed_seed_sweep(
    X: np.ndarray,
    y: np.ndarray,
    seed_plan: SeedPlan,
    settings: Settings,
    *,
    output_dir: Path,
    logger: Any,
    resume: bool,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    n, runs = len(y), settings.n_visualization_runs
    shape = (n, runs)
    predictions: dict[str, np.ndarray] = {}
    for model in MODEL_KEYS:
        path = _observed_path(output_dir, model)
        predictions[model] = (
            _load_array(path, shape, allow_nan=True)
            if resume and path.exists()
            else np.full(shape, np.nan, dtype=float)
        )
    complete = np.stack([np.isfinite(values).all(axis=0) for values in predictions.values()])
    for column in range(runs):
        if complete[:, column].any() and not complete[:, column].all():
            raise ValueError(f"observed checkpoint has partial seed column {column}")
    pending = np.flatnonzero(~complete.all(axis=0)).tolist()
    issues = _load_issue_counts(
        output_dir / "compass_observed_issue_counts.json", resume=resume
    )
    if not pending:
        return predictions, dict(issues)

    workers = resolve_worker_count(settings.n_jobs, n_tasks=len(pending))
    batch = (
        min(len(pending), settings.observed_batch_size)
        if settings.observed_batch_size
        else workers
    )
    seeds = seed_plan.joint[:runs]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        for start in range(0, len(pending), batch):
            columns = pending[start : start + batch]
            results = Parallel(n_jobs=workers, batch_size=1)(
                delayed(_evaluate_observed_seed)(
                    column, int(seeds[column]), X, y, settings
                )
                for column in columns
            )
            for column, cell_predictions, cell_issues in results:
                for model, values in cell_predictions.items():
                    predictions[model][:, column] = values
                issues.update(cell_issues)
            _save_observed_state(output_dir, predictions, issues)
            completed = int(
                np.stack(
                    [np.isfinite(values).all(axis=0) for values in predictions.values()]
                ).all(axis=0).sum()
            )
            logger.info("Observed-data sweep: %d/%d seed vectors complete", completed, runs)
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise RuntimeError("observed COMPAS probability matrices are incomplete")
    return predictions, dict(issues)


def per_uid_stats(oof: np.ndarray, threshold: float):
    """Return the legacy per-person instability statistics.

    The complete seven-value return signature is retained for notebooks that
    imported the original builder directly, even though the production summary
    currently writes only the mean, standard deviation, and flip rate.
    """

    values = np.asarray(oof, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("OOF matrix must be finite with at least two seed columns")
    mean = values.mean(axis=1)
    sd = values.std(axis=1, ddof=1)
    q10 = np.percentile(values, 10, axis=1)
    q50 = np.percentile(values, 50, axis=1)
    q90 = np.percentile(values, 90, axis=1)
    binary = values >= threshold
    ones = binary.sum(axis=1)
    flips = np.minimum(ones, values.shape[1] - ones)
    return mean, sd, q10, q50, q90, flips, flips / values.shape[1]


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Retain the legacy finite sample mean/SD helper."""

    array = np.asarray(values, dtype=float)
    return float(np.nanmean(array)), float(np.nanstd(array, ddof=1))


def _observed_scores(
    y: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    threshold: float,
) -> dict[str, np.ndarray]:
    runs = next(iter(predictions.values())).shape[1]
    scores = {key: np.empty(runs, dtype=float) for key in SCORE_KEYS}
    for column in range(runs):
        cell = {model: predictions[model][:, column] for model in MODEL_KEYS}
        cell_scores = _scores_from_predictions(y, cell, threshold)
        for key, value in cell_scores.items():
            scores[key][column] = value
    return scores


def _estimand(key: str, settings: Settings) -> str:
    model, metric = key.rsplit("_", 1)
    return (
        f"COMPAS {MODEL_LABELS[model]} {settings.n_splits}-fold OOF "
        f"{METRIC_LABELS[metric]}"
    )


def _score_provenance(
    *,
    settings: Settings,
    data_path: Path,
    seed_path: Path,
    seed_plan: SeedPlan,
) -> dict[str, Any]:
    analysis_dir = Path(__file__).resolve().parent
    score_settings = {
        key: value
        for key, value in asdict(settings).items()
        if key not in {"n_jobs", "bootstrap_batch_size", "observed_batch_size"}
    }
    return {
        "schema_version": 1,
        "score_settings": score_settings,
        "data_sha256": _sha256(data_path),
        "seed_list_sha256": _sha256(seed_path),
        "joint_seeds": seed_plan.joint.tolist(),
        "bootstrap_seeds": seed_plan.bootstrap.tolist(),
        "builder_sha256": _sha256(Path(__file__).resolve()),
        "reporting_utils_sha256": _sha256(analysis_dir / "reporting_utils.py"),
        "versions": {
            "Python": platform.python_version(),
            "NumPy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }


def _prepare_provenance(path: Path, payload: Mapping[str, Any], *, require_match: bool) -> None:
    safe = _json_safe(payload)
    if require_match:
        if not path.exists():
            raise ValueError("resume/reuse requires an existing score-provenance file")
        with path.open(encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved != safe:
            raise ValueError(
                "saved COMPAS score provenance does not match this run; start fresh"
            )
    else:
        _atomic_write_json(path, safe)


def write_outputs(
    *,
    output_dir: Path,
    uid: np.ndarray,
    X: pd.DataFrame,
    y: np.ndarray,
    compas_decile: np.ndarray,
    compas_hat: np.ndarray,
    bootstrap_indices: np.ndarray,
    seed_plan: SeedPlan,
    settings: Settings,
    crossed: Mapping[str, np.ndarray],
    observed_predictions: Mapping[str, np.ndarray],
    observed_scores: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_save_array(output_dir / "compass_bootstrap_indices.npy", bootstrap_indices)
    for key, values in crossed.items():
        _atomic_save_array(_score_path(output_dir, key), values)
    for model, values in observed_predictions.items():
        _atomic_save_array(_observed_path(output_dir, model), values)

    B, m = settings.n_bootstrap_resamples, settings.n_seed_vectors
    bootstrap_number = np.repeat(np.arange(B), m)
    seed_number = np.tile(np.arange(m), B)
    crossed_frame = pd.DataFrame(
        {
            "Bootstrap_Replicate": bootstrap_number,
            "Seed_Vector": seed_number,
            "Bootstrap_Seed": seed_plan.bootstrap[bootstrap_number],
            "Joint_Folding_Model_Seed": seed_plan.joint[seed_number],
            **{key: values.ravel() for key, values in crossed.items()},
        }
    )
    crossed_frame.to_csv(
        output_dir / "compass_crossed_bootstrap_scores.csv", index=False
    )

    runs = settings.n_visualization_runs
    observed_frame = pd.DataFrame(
        {
            "Seed_Vector": np.arange(runs),
            "Joint_Folding_Model_Seed": seed_plan.joint[:runs],
            **observed_scores,
        }
    )
    observed_frame.to_csv(output_dir / "compass_visualization_runs.csv", index=False)

    prediction_columns: dict[str, Any] = {
        "y": y,
        "compas_decile": compas_decile,
        "compas_hat": compas_hat,
    }
    for model in ("rf", "lr", "nn"):
        for column, seed in enumerate(seed_plan.joint[:runs]):
            prediction_columns[f"y_hat_{model}_seed{int(seed)}"] = (
                observed_predictions[model][:, column]
            )
    prediction_frame = pd.DataFrame(prediction_columns, index=uid)
    prediction_frame.index.name = "UID"
    prediction_name = (
        f"uid_oof_predictions_{runs}seeds_rf_lr_nn_compas_"
        f"{settings.n_splits}folds.csv"
    )
    prediction_frame.to_csv(output_dir / prediction_name)

    summary_columns: dict[str, Any] = {"y": y}
    for model in ("rf", "lr", "nn"):
        mean, sd, *_unused, flip_rate = per_uid_stats(
            observed_predictions[model], settings.threshold
        )
        summary_columns.update(
            {
                f"{model}_mu": mean,
                f"{model}_sd": sd,
                f"{model}_fliprate": flip_rate,
            }
        )
    summary_columns.update(
        {
            "age": X["age"].to_numpy(),
            "priors": X["priors_count"].to_numpy(),
            "compas_decile": compas_decile,
        }
    )
    summary = pd.DataFrame(summary_columns, index=uid)
    summary.index.name = "UID"
    summary.to_csv(
        output_dir
        / (
            f"uid_summary_instability_{runs}seeds_{settings.n_splits}folds_"
            "rf_lr_nn.csv"
        )
    )

    pd.DataFrame(
        {
            "Bootstrap_Replicate": np.arange(B),
            "Bootstrap_Seed": seed_plan.bootstrap[:B],
        }
    ).to_csv(output_dir / "compass_bootstrap_plan.csv", index=False)
    pd.DataFrame(
        {
            "Seed_Vector": np.arange(len(seed_plan.joint)),
            "Joint_Folding_Model_Seed": seed_plan.joint,
        }
    ).to_csv(output_dir / "compass_seed_plan.csv", index=False)
    pd.DataFrame(
        [
            {"Estimand": _estimand(key, settings), **diagnostics[key].to_dict()}
            for key in SCORE_KEYS
        ]
    ).to_csv(output_dir / "compass_s5_diagnostics.csv", index=False)

    payload = {
        "metadata": metadata,
        "diagnostics": {
            _estimand(key, settings): diagnostics[key].to_dict()
            for key in SCORE_KEYS
        },
        "observed_data_seed_summary": {
            _estimand(key, settings): observed_seed_summary(values).to_dict()
            for key, values in observed_scores.items()
        },
    }
    _atomic_write_json(output_dir / "compass_s5_diagnostics.json", payload)


def _build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run crossed COMPAS S5 diagnostics and a separate 1,000-seed "
            "observed-data OOF-probability sweep."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=repo_root / "data/compass/raw/compas-scores-two-years.csv",
    )
    parser.add_argument("--data-url", default=DATA_URL)
    parser.add_argument(
        "--seed-list", type=Path, default=repo_root / "assets/seed_list.txt"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=repo_root / "data/compass/results"
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=repo_root / "results/results_logs/compass_s5_diagnostics.log",
    )
    parser.add_argument("--n-bootstrap-resamples", type=int, default=100)
    parser.add_argument("--n-seed-vectors", type=int, default=100)
    parser.add_argument("--n-visualization-runs", type=int, default=1_000)
    parser.add_argument("--n-splits", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--rf-n-estimators", type=int, default=200)
    parser.add_argument("--rf-max-depth", type=int, default=5)
    parser.add_argument("--model-max-iter", type=int, default=2_000)
    parser.add_argument("--mlp-hidden-units", type=int, default=16)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--bootstrap-batch-size",
        type=int,
        default=0,
        help="bootstrap rows per checkpoint; 0 uses one worker wave",
    )
    parser.add_argument("--observed-batch-size", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reuse-crossed-scores",
        action="store_true",
        help="reuse complete crossed score arrays after provenance validation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="2x2 crossed grid, four visualization seeds, and smaller models",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    started = time.monotonic()
    repo_root = Path(__file__).resolve().parents[2]
    args = _build_parser(repo_root).parse_args(argv)
    if args.dry_run:
        args.n_bootstrap_resamples = 2
        args.n_seed_vectors = 2
        args.n_visualization_runs = 4
        args.rf_n_estimators = min(args.rf_n_estimators, 5)
        args.model_max_iter = min(args.model_max_iter, 100)
        args.observed_batch_size = 4
    settings = Settings(
        n_bootstrap_resamples=args.n_bootstrap_resamples,
        n_seed_vectors=args.n_seed_vectors,
        n_visualization_runs=args.n_visualization_runs,
        n_splits=args.n_splits,
        threshold=args.threshold,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
        model_max_iter=args.model_max_iter,
        mlp_hidden_units=args.mlp_hidden_units,
        n_jobs=args.n_jobs,
        bootstrap_batch_size=args.bootstrap_batch_size,
        observed_batch_size=args.observed_batch_size,
    )
    settings.validate()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = _ensure_input(args.data_path.resolve(), args.data_url)
    _df, X_frame, y, uid, compas_decile, compas_hat = load_and_preprocess(data_path)
    if settings.n_splits > len(y):
        raise ValueError("n_splits exceeds the filtered cohort size")
    X = np.ascontiguousarray(X_frame.to_numpy(dtype=float))
    y = np.ascontiguousarray(y, dtype=np.int8)
    seeds = load_seed_list(args.seed_list)
    seed_plan = make_seed_plan(seeds, settings)
    bootstrap_indices = bootstrap_index_block(
        len(y), seed_plan.bootstrap[: settings.n_bootstrap_resamples]
    )
    provenance = _score_provenance(
        settings=settings,
        data_path=data_path,
        seed_path=args.seed_list.resolve(),
        seed_plan=seed_plan,
    )
    _prepare_provenance(
        output_dir / "compass_score_provenance.json",
        provenance,
        require_match=bool(args.resume or args.reuse_crossed_scores),
    )

    pending_log = args.log_path.resolve().with_suffix(".pending.log")
    logger = configure_run_logger("compass_s5", pending_log)
    workers = resolve_worker_count(settings.n_jobs)
    logger.info("Starting COMPAS S5 analysis")
    logger.info("Filtered cohort: %d observations; features: age, priors_count", len(y))
    logger.info("Input: %s (sha256=%s)", data_path, _sha256(data_path))
    logger.info("Seed list: %s", args.seed_list.resolve())
    logger.info("Output directory: %s", output_dir)
    logger.info("Outer joblib workers: %d; inner estimator workers/BLAS threads: 1", workers)
    logger.info(
        "Prepared %d shared bootstrap datasets x %d joint seed vectors (%d cells)",
        settings.n_bootstrap_resamples,
        settings.n_seed_vectors,
        settings.n_bootstrap_resamples * settings.n_seed_vectors,
    )
    _atomic_save_array(output_dir / "compass_bootstrap_indices.npy", bootstrap_indices)

    shape = (settings.n_bootstrap_resamples, settings.n_seed_vectors)
    if args.reuse_crossed_scores:
        crossed = {
            key: _load_array(_score_path(output_dir, key), shape, allow_nan=False)
            for key in SCORE_KEYS
        }
        issue_path = output_dir / "compass_crossed_issue_counts.json"
        crossed_issues = dict(_load_issue_counts(issue_path, resume=True))
        logger.info("Reused all %d crossed cells", shape[0] * shape[1])
    else:
        crossed, crossed_issues = run_crossed_grid(
            X,
            y,
            bootstrap_indices,
            seed_plan,
            settings,
            output_dir=output_dir,
            logger=logger,
            resume=args.resume,
        )
    observed_predictions, observed_issues = run_observed_seed_sweep(
        X,
        y,
        seed_plan,
        settings,
        output_dir=output_dir,
        logger=logger,
        resume=args.resume,
    )
    observed_scores = _observed_scores(y, observed_predictions, settings.threshold)
    diagnostics = {
        key: crossed_s5_diagnostics(values) for key, values in crossed.items()
    }

    auc_compas = float(roc_auc_score(y, compas_decile))
    acc_compas = float(accuracy_score(y, compas_hat))
    common_details = {
        "within_cell_replications": "1 complete two-fold OOF evaluation",
        "external_dataset_bootstrap": (
            "ordinary n-out-of-n row bootstrap of the filtered cohort"
        ),
        "bootstrap_design": (
            "crossed; each stored bootstrap index row is reused for every "
            "joint seed vector and all three models"
        ),
        "bootstrap_PRNG": f"NumPy PCG64 {np.__version__}",
        "algorithmic_PRNG": (
            "one asset-list integer passed to shuffled KFold and every "
            "scikit-learn model random_state"
        ),
        "seed_components": "joint fold-assignment and model-training seed",
        "seed_components_varied": "jointly, matching manuscript S3.8",
        "cross_validation": f"{settings.n_splits}-fold non-stratified shuffled KFold",
        "features": "age and priors_count",
        "classification_threshold": settings.threshold,
        "models": (
            f"RF(n_estimators={settings.rf_n_estimators},max_depth={settings.rf_max_depth}); "
            f"LR(saga,L2,max_iter={settings.model_max_iter}); "
            f"MLP(hidden={settings.mlp_hidden_units},relu,lbfgs,standardized,"
            f"max_iter={settings.model_max_iter})"
        ),
        "fit_failure_policy": "fail loudly; no prediction fallback or substitution",
        "crossed_convergence_warnings": crossed_issues.get("convergence_warnings", 0),
        "crossed_other_warnings": crossed_issues.get("other_warnings", 0),
        "Python_version": platform.python_version(),
        "NumPy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "parallel_backend": "joblib loky bootstrap-row tasks; estimator n_jobs=1",
        "parallel_workers": workers,
    }
    for key in SCORE_KEYS:
        estimand = _estimand(key, settings)
        logger.info(
            "\n%s",
            format_s5_report(
                diagnostics[key],
                estimand=estimand,
                computational_details=common_details,
            ),
        )
        logger.info(
            "\n%s",
            format_observed_seed_report(
                observed_seed_summary(observed_scores[key]), estimand=estimand
            ),
        )
    logger.info(
        "Seed-invariant COMPAS benchmark: AUC=%.12g, accuracy(decile>=5)=%.12g",
        auc_compas,
        acc_compas,
    )
    logger.info(
        "Observed sweep warning counts: convergence=%d, other=%d",
        observed_issues.get("convergence_warnings", 0),
        observed_issues.get("other_warnings", 0),
    )

    metadata = {
        "analysis": "ProPublica COMPAS recidivism seed sensitivity",
        "settings": asdict(settings),
        "estimands": [_estimand(key, settings) for key in SCORE_KEYS],
        "n_filtered_observations": len(y),
        "data_source_url": args.data_url,
        "data_path": str(data_path),
        "data_sha256": provenance["data_sha256"],
        "seed_list": str(args.seed_list.resolve()),
        "seed_list_sha256": provenance["seed_list_sha256"],
        "crossed_scores_reused": bool(args.reuse_crossed_scores),
        "resumed_from_checkpoints": bool(args.resume),
        "crossed_issue_counts": crossed_issues,
        "observed_issue_counts": observed_issues,
        "compas_seed_invariant_benchmark": {
            "auc": auc_compas,
            "accuracy_decile_at_least_5": acc_compas,
        },
        "computational_details": common_details,
        "provenance": provenance,
    }
    write_outputs(
        output_dir=output_dir,
        uid=uid,
        X=X_frame,
        y=y,
        compas_decile=compas_decile,
        compas_hat=compas_hat,
        bootstrap_indices=bootstrap_indices,
        seed_plan=seed_plan,
        settings=settings,
        crossed=crossed,
        observed_predictions=observed_predictions,
        observed_scores=observed_scores,
        diagnostics=diagnostics,
        metadata=metadata,
    )
    logger.info(
        "COMPAS analysis complete in %.1f seconds; all outputs and six S5 "
        "report blocks written",
        time.monotonic() - started,
    )
    for handler in logger.handlers:
        handler.flush()
    estimands = tuple(_estimand(key, settings) for key in SCORE_KEYS)
    validate_s5_log(pending_log, estimands)
    args.log_path.resolve().parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pending_log, args.log_path.resolve())
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    pending_log.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
