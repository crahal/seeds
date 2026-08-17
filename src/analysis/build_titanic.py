"""Run the Titanic seed and sampling-uncertainty analysis.

The S5 experiment crosses one shared block of nonparametric bootstrap datasets
with paired algorithmic seed vectors. A separate no-bootstrap seed sweep
supplies the visualisations and is never used for S5 variance components.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import statsmodels.api as sm
from joblib import Parallel, delayed, parallel_config
from scipy.optimize import brentq
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold

try:  # Package import and direct execution are both supported.
    from .reporting_utils import (
        S5Diagnostics,
        bootstrap_index_block,
        configure_run_logger,
        crossed_s5_diagnostics,
        format_observed_seed_report,
        format_s5_report,
        full_factorial_seed_grid,
        load_seed_list,
        observed_seed_summary,
        parallel_chunk_ranges,
        resolve_batch_size,
        resolve_worker_count,
        seed_component_blocks,
        validate_s5_log,
    )
except ImportError:  # pragma: no cover - direct CLI execution.
    from reporting_utils import (  # type: ignore[no-redef]
        S5Diagnostics,
        bootstrap_index_block,
        configure_run_logger,
        crossed_s5_diagnostics,
        format_observed_seed_report,
        format_s5_report,
        full_factorial_seed_grid,
        load_seed_list,
        observed_seed_summary,
        parallel_chunk_ranges,
        resolve_batch_size,
        resolve_worker_count,
        seed_component_blocks,
        validate_s5_log,
    )


TARGET = "Survived"
FEATURE_COLUMNS = (
    "Pclass", "Sex", "Age", "Fare", "Embarked", "Title", "IsAlone", "Age*Class"
)
TITLE_CODES = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
SCORE_KEYS = (
    "LogisticRegression_R2", "LogisticRegression_IMV", "SGD_R2", "SGD_IMV"
)
SCORE_FILE_STEMS = {
    "LogisticRegression_R2": "titanic_crossed_logistic_r2_scores",
    "LogisticRegression_IMV": "titanic_crossed_logistic_imv_scores",
    "SGD_R2": "titanic_crossed_sgd_r2_scores",
    "SGD_IMV": "titanic_crossed_sgd_imv_scores",
}
ISSUE_NAMES = (
    "logistic_fit_fallbacks",
    "logistic_fit_warnings",
    "logistic_metric_substitutions",
    "sgd_fit_fallbacks",
    "sgd_fit_warnings",
    "sgd_metric_substitutions",
)


@dataclass(frozen=True)
class Settings:
    n_bootstrap_resamples: int = 100
    n_seed_vectors: int = 100
    n_visualization_runs: int = 1_000
    n_jobs: int = -1
    bootstrap_batch_size: int = 0
    n_folds: int = 5
    sgd_max_iter: int = 1_000
    sgd_tolerance: float = 1e-4
    logistic_max_iter: int = 250

    def validate(self) -> None:
        if self.n_bootstrap_resamples < 2 or self.n_seed_vectors < 2:
            raise ValueError("S5 diagnostics require at least two bootstraps and seeds")
        if self.n_visualization_runs < 2:
            raise ValueError("n_visualization_runs must be at least two")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")
        resolve_worker_count(self.n_jobs)
        if self.bootstrap_batch_size < 0:
            raise ValueError("bootstrap_batch_size cannot be negative")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least two")
        if self.sgd_max_iter < 1 or self.logistic_max_iter < 1:
            raise ValueError("model iteration limits must be positive")
        if self.sgd_tolerance <= 0:
            raise ValueError("sgd_tolerance must be positive")


@dataclass(frozen=True)
class SeedPlan:
    folding: np.ndarray
    modeling: np.ndarray
    bootstrap: np.ndarray


@dataclass(frozen=True)
class ModelResult:
    r2: float
    imv: float
    accuracy: float
    fit_fallbacks: int
    fit_warnings: int
    metric_substitutions: int


@dataclass(frozen=True)
class CellResult:
    logistic: ModelResult
    sgd: ModelResult


@dataclass(frozen=True)
class ObservedSweep:
    logistic_r2: np.ndarray
    logistic_imv: np.ndarray
    sgd_r2: np.ndarray
    sgd_imv: np.ndarray
    sgd_folding: np.ndarray
    sgd_modeling: np.ndarray
    issue_counts: np.ndarray


def read_titanic_data(csv_path: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    required = {
        TARGET, "PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp",
        "Parch", "Ticket", "Fare", "Cabin", "Embarked",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise KeyError(f"Titanic data are missing required columns: {missing}")
    if len(data) < 2:
        raise ValueError("Titanic data must contain at least two observations")
    return data


def wrangle_titanic(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Apply the legacy feature construction robustly to a supplied dataset."""

    data = train_df.copy(deep=True).reset_index(drop=True)
    required = {
        TARGET, "PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp",
        "Parch", "Ticket", "Fare", "Cabin", "Embarked",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise KeyError(f"Titanic data are missing required columns: {missing}")
    data = data.drop(columns=["Ticket", "Cabin"])

    titles = data["Name"].astype("string").str.extract(
        r" ([A-Za-z]+)\.", expand=False
    )
    titles = titles.replace(
        ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev",
         "Sir", "Jonkheer", "Dona"],
        "Rare",
    ).replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    data["Title"] = titles.map(TITLE_CODES).fillna(0).astype(int)
    data = data.drop(columns=["Name", "PassengerId"])

    sex = data["Sex"].map({"female": 1, "male": 0})
    if sex.isna().any():
        unknown = sorted(data.loc[sex.isna(), "Sex"].astype(str).unique())
        raise ValueError(f"unknown Sex values: {unknown}")
    data["Sex"] = sex.astype(int)

    # Force floating storage before half-year imputation; bootstrap fixtures can
    # otherwise retain an integer dtype that rejects values such as 28.5.
    data["Age"] = pd.to_numeric(data["Age"], errors="coerce").astype(float)
    overall_age = float(data["Age"].median())
    if not np.isfinite(overall_age):
        raise ValueError("at least one finite Age is required for imputation")
    for sex_code in (0, 1):
        for passenger_class in (1, 2, 3):
            stratum = data.loc[
                (data["Sex"] == sex_code) & (data["Pclass"] == passenger_class),
                "Age",
            ].dropna()
            median = overall_age if stratum.empty else float(stratum.median())
            rounded_half_year = np.floor(median / 0.5 + 0.5) * 0.5
            mask = (
                data["Age"].isna()
                & (data["Sex"] == sex_code)
                & (data["Pclass"] == passenger_class)
            )
            data.loc[mask, "Age"] = rounded_half_year
    data["Age"] = data["Age"].fillna(overall_age).astype(int)
    raw_age = data["Age"].to_numpy(copy=True)
    data["Age"] = np.select(
        [raw_age <= 16, (raw_age > 16) & (raw_age <= 32),
         (raw_age > 32) & (raw_age <= 48),
         (raw_age > 48) & (raw_age <= 64), raw_age > 64],
        [0, 1, 2, 3, 5],
    ).astype(int)

    family_size = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (family_size == 1).astype(int)
    data = data.drop(columns=["Parch", "SibSp"])
    data["Age*Class"] = data["Age"] * data["Pclass"]

    embarked_mode = data["Embarked"].dropna().mode()
    embarked_fill = "S" if embarked_mode.empty else str(embarked_mode.iloc[0])
    embarked = data["Embarked"].fillna(embarked_fill).map({"S": 0, "C": 1, "Q": 2})
    if embarked.isna().any():
        unknown = sorted(data.loc[embarked.isna(), "Embarked"].dropna().astype(str).unique())
        raise ValueError(f"unknown Embarked values: {unknown}")
    data["Embarked"] = embarked.astype(int)

    fare = pd.to_numeric(data["Fare"], errors="coerce")
    fare_median = float(fare.median())
    if not np.isfinite(fare_median):
        raise ValueError("at least one finite Fare is required")
    fare = fare.fillna(fare_median).to_numpy()
    data["Fare"] = np.select(
        [fare <= 7.91, (fare > 7.91) & (fare <= 14.454),
         (fare > 14.454) & (fare <= 31), fare > 31],
        [0, 1, 2, 3],
    ).astype(int)

    target = pd.to_numeric(data[TARGET], errors="raise").astype(int)
    if not set(target.unique()).issubset({0, 1}):
        raise ValueError("Survived must be binary (0/1)")
    features = data.loc[:, FEATURE_COLUMNS].astype(np.float64)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("wrangled Titanic predictors must be complete and finite")
    return features, target


def make_seed_plan(seed_list: Sequence[int], settings: Settings) -> SeedPlan:
    n_vectors = max(settings.n_seed_vectors, settings.n_visualization_runs)
    components = seed_component_blocks(
        seed_list, n_vectors=n_vectors, component_names=("folding", "modeling")
    )
    bootstrap_start = 2 * n_vectors
    bootstrap_stop = bootstrap_start + settings.n_bootstrap_resamples
    if bootstrap_stop > len(seed_list):
        raise ValueError(
            f"seed list has {len(seed_list)} entries but {bootstrap_stop} are needed"
        )
    return SeedPlan(
        folding=components["folding"],
        modeling=components["modeling"],
        bootstrap=np.asarray(seed_list[bootstrap_start:bootstrap_stop], dtype=np.uint64),
    )


def observed_sgd_seed_grid(
    seed_plan: SeedPlan, n_runs: int
) -> tuple[np.ndarray, np.ndarray, int, int]:
    return full_factorial_seed_grid(
        seed_plan.folding, seed_plan.modeling, n_runs=n_runs
    )


def _as_arrays(
    features: pd.DataFrame | np.ndarray, target: pd.Series | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.int64)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("features and target have incompatible shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("features and target must be finite")
    return x, y


_ENTROPY_WEIGHT_CACHE: dict[float, float] = {}
_ENTROPY_WEIGHT_CACHE_LIMIT = 8_192


def _solve_entropy_weight(likelihood: float) -> float:
    """Invert binary entropy on the upper branch used by the legacy IMV.

    The previous implementation asked a general-purpose multivariate
    minimizer to solve this monotone scalar equation. A bounded root solve is
    mathematically equivalent, hundreds of times faster, and more accurate.
    """

    value = float(likelihood)
    if not np.isfinite(value) or value <= 0:
        return float("nan")

    lower = 0.5
    upper = 0.999

    def entropy(probability: float) -> float:
        return probability * math.log(probability) + (
            1 - probability
        ) * math.log1p(-probability)

    target = math.log(value)
    lower_entropy = entropy(lower)
    upper_entropy = entropy(upper)
    if target <= lower_entropy:
        return lower
    if target >= upper_entropy:
        return upper
    try:
        return float(
            brentq(
                lambda probability: entropy(probability) - target,
                lower,
                upper,
                xtol=1e-12,
                rtol=1e-14,
            )
        )
    except (ValueError, RuntimeError):
        return float("nan")


def _entropy_weight(likelihood: float) -> float:
    """Return a bounded entropy inverse from a process-local result cache."""

    key = float(likelihood)
    cached = _ENTROPY_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    value = _solve_entropy_weight(key)
    if np.isfinite(value):
        if len(_ENTROPY_WEIGHT_CACHE) >= _ENTROPY_WEIGHT_CACHE_LIMIT:
            _ENTROPY_WEIGHT_CACHE.clear()
        _ENTROPY_WEIGHT_CACHE[key] = value
    return value


def calculate_scores(
    y_test: np.ndarray, probabilities: np.ndarray, y_train: np.ndarray
) -> tuple[float, float, int]:
    """Calculate legacy pseudo-R2 and IMV, counting nonfinite substitutions."""

    epsilon = 1e-6
    truth = np.asarray(y_test, dtype=float)
    probability = np.asarray(probabilities, dtype=float).copy()
    train = np.asarray(y_train, dtype=float)
    probability[probability == 0] += 0.0001
    probability[probability == 1] -= 0.001
    probability = np.clip(probability, epsilon, 1 - epsilon)
    substitutions = 0

    denominator = np.square(truth - train.mean()).sum()
    r2 = 1 - np.square(truth - probability).sum() / denominator
    if not np.isfinite(r2):
        r2 = 0.0
        substitutions += 1

    def geometric_likelihood(values: np.ndarray | float) -> float:
        prediction = np.clip(values, epsilon, 1 - epsilon)
        terms = np.log(prediction) * truth + np.log1p(-prediction) * (1 - truth)
        return float(max(epsilon, np.exp(np.mean(terms))))

    w0 = _entropy_weight(geometric_likelihood(float(train.mean())))
    w1 = _entropy_weight(geometric_likelihood(probability))
    imv = (
        float("nan")
        if not np.isfinite(w0) or not np.isfinite(w1) or w0 == 0
        else (w1 - w0) / w0
    )
    if not np.isfinite(imv):
        imv = 0.0
        substitutions += 1
    return float(r2), float(imv), substitutions


def _logistic_probabilities(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, settings: Settings
) -> tuple[np.ndarray, int, int]:
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = sm.Logit(y_train, sm.add_constant(x_train, has_constant="add"))
            result = model.fit(disp=False, maxiter=settings.logistic_max_iter)
        probabilities = np.asarray(
            result.predict(sm.add_constant(x_test, has_constant="add")), dtype=float
        )
        if probabilities.shape != (len(x_test),) or not np.isfinite(probabilities).all():
            raise ValueError("Logit returned invalid probabilities")
        return probabilities, 0, len(caught)
    except Exception:
        return np.full(len(x_test), y_train.mean(), dtype=float), 1, 0


def _sgd_probabilities(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    modeling_seed: int,
    settings: Settings,
) -> tuple[np.ndarray, int, int]:
    try:
        classifier = SGDClassifier(
            loss="log_loss",
            max_iter=settings.sgd_max_iter,
            tol=settings.sgd_tolerance,
            random_state=int(modeling_seed),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(x_train, y_train)
        probabilities = np.asarray(classifier.predict_proba(x_test)[:, 1], dtype=float)
        if probabilities.shape != (len(x_test),) or not np.isfinite(probabilities).all():
            raise ValueError("SGD returned invalid probabilities")
        return probabilities, 0, len(caught)
    except Exception:
        return np.full(len(x_test), y_train.mean(), dtype=float), 1, 0


def _evaluate_model_on_splits(
    features: np.ndarray,
    target: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    model: str,
    modeling_seed: int,
    settings: Settings,
) -> ModelResult:
    r2_values: list[float] = []
    imv_values: list[float] = []
    accuracies: list[float] = []
    fit_fallbacks = fit_warnings = metric_substitutions = 0
    for train_index, test_index in splits:
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = target[train_index], target[test_index]
        if model == "logistic":
            probability, fallback, warning_count = _logistic_probabilities(
                x_train, y_train, x_test, settings
            )
        elif model == "sgd":
            probability, fallback, warning_count = _sgd_probabilities(
                x_train, y_train, x_test,
                modeling_seed=modeling_seed, settings=settings,
            )
        else:  # pragma: no cover
            raise ValueError(f"unknown model: {model}")
        r2, imv, substitutions = calculate_scores(y_test, probability, y_train)
        r2_values.append(r2)
        imv_values.append(imv)
        accuracies.append(float(np.mean((probability >= 0.5) == y_test)))
        fit_fallbacks += fallback
        fit_warnings += warning_count
        metric_substitutions += substitutions
    return ModelResult(
        r2=float(np.mean(r2_values)),
        imv=float(np.mean(imv_values)),
        accuracy=float(np.mean(accuracies)),
        fit_fallbacks=fit_fallbacks,
        fit_warnings=fit_warnings,
        metric_substitutions=metric_substitutions,
    )


_KFOLD_SPLIT_CACHE: dict[
    tuple[int, int, int], tuple[tuple[np.ndarray, np.ndarray], ...]
] = {}
_KFOLD_SPLIT_CACHE_LIMIT = 512


def _cached_kfold_splits(
    n_observations: int, n_folds: int, folding_seed: int
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Cache positional folds, which are data-independent at fixed row count."""

    key = (int(n_observations), int(n_folds), int(folding_seed))
    cached = _KFOLD_SPLIT_CACHE.get(key)
    if cached is not None:
        return cached
    splitter = KFold(
        n_splits=n_folds, shuffle=True, random_state=int(folding_seed)
    )
    splits = tuple(splitter.split(np.empty(n_observations)))
    for train_index, test_index in splits:
        train_index.flags.writeable = False
        test_index.flags.writeable = False
    if len(_KFOLD_SPLIT_CACHE) >= _KFOLD_SPLIT_CACHE_LIMIT:
        _KFOLD_SPLIT_CACHE.clear()
    _KFOLD_SPLIT_CACHE[key] = splits
    return splits


def _make_splits(
    features: np.ndarray, target: np.ndarray, *, folding_seed: int, settings: Settings
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if settings.n_folds > len(target):
        raise ValueError("n_folds cannot exceed the number of observations")
    return _cached_kfold_splits(
        len(target), settings.n_folds, int(folding_seed)
    )


def _evaluate_models_arrays(
    features: np.ndarray,
    target: np.ndarray,
    *,
    folding_seed: int,
    modeling_seed: int,
    settings: Settings,
) -> CellResult:
    splits = _make_splits(
        features, target, folding_seed=folding_seed, settings=settings
    )
    return CellResult(
        logistic=_evaluate_model_on_splits(
            features,
            target,
            splits,
            model="logistic",
            modeling_seed=modeling_seed,
            settings=settings,
        ),
        sgd=_evaluate_model_on_splits(
            features,
            target,
            splits,
            model="sgd",
            modeling_seed=modeling_seed,
            settings=settings,
        ),
    )


def _evaluate_logistic_arrays(
    features: np.ndarray,
    target: np.ndarray,
    *,
    folding_seed: int,
    settings: Settings,
) -> ModelResult:
    splits = _make_splits(
        features, target, folding_seed=folding_seed, settings=settings
    )
    return _evaluate_model_on_splits(
        features,
        target,
        splits,
        model="logistic",
        modeling_seed=0,
        settings=settings,
    )


def _evaluate_sgd_arrays(
    features: np.ndarray,
    target: np.ndarray,
    *,
    folding_seed: int,
    modeling_seed: int,
    settings: Settings,
) -> ModelResult:
    splits = _make_splits(
        features, target, folding_seed=folding_seed, settings=settings
    )
    return _evaluate_model_on_splits(
        features,
        target,
        splits,
        model="sgd",
        modeling_seed=modeling_seed,
        settings=settings,
    )


def evaluate_models(
    features: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    *,
    folding_seed: int,
    modeling_seed: int,
    settings: Settings,
) -> CellResult:
    x, y = _as_arrays(features, target)
    return _evaluate_models_arrays(
        x,
        y,
        folding_seed=folding_seed,
        modeling_seed=modeling_seed,
        settings=settings,
    )


def evaluate_logistic(
    features: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    *,
    folding_seed: int,
    settings: Settings,
) -> ModelResult:
    x, y = _as_arrays(features, target)
    return _evaluate_logistic_arrays(
        x, y, folding_seed=folding_seed, settings=settings
    )


def evaluate_sgd(
    features: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    *,
    folding_seed: int,
    modeling_seed: int,
    settings: Settings,
) -> ModelResult:
    x, y = _as_arrays(features, target)
    return _evaluate_sgd_arrays(
        x,
        y,
        folding_seed=folding_seed,
        modeling_seed=modeling_seed,
        settings=settings,
    )


def _issue_vector(result: CellResult) -> np.ndarray:
    return np.asarray(
        [
            result.logistic.fit_fallbacks,
            result.logistic.fit_warnings,
            result.logistic.metric_substitutions,
            result.sgd.fit_fallbacks,
            result.sgd.fit_warnings,
            result.sgd.metric_substitutions,
        ],
        dtype=np.int64,
    )


def _single_issue_vector(result: ModelResult, *, model: str) -> np.ndarray:
    counts = np.zeros(len(ISSUE_NAMES), dtype=np.int64)
    offset = 0 if model == "logistic" else 3
    counts[offset : offset + 3] = (
        result.fit_fallbacks,
        result.fit_warnings,
        result.metric_substitutions,
    )
    return counts


def _evaluate_bootstrap_row(
    bootstrap_number: int,
    row_indices: np.ndarray,
    raw_data: pd.DataFrame,
    folding_seeds: np.ndarray,
    modeling_seeds: np.ndarray,
    settings: Settings,
) -> tuple[int, dict[str, np.ndarray], np.ndarray]:
    """Wrangle one raw resample once, then reuse it for every seed vector."""

    sampled = raw_data.iloc[row_indices].reset_index(drop=True)
    features, target = wrangle_titanic(sampled)
    x, y = _as_arrays(features, target)
    row_scores = {
        key: np.empty(len(folding_seeds), dtype=float) for key in SCORE_KEYS
    }
    issues = np.zeros(len(ISSUE_NAMES), dtype=np.int64)
    for seed_number, (folding_seed, modeling_seed) in enumerate(
        zip(folding_seeds, modeling_seeds, strict=True)
    ):
        result = _evaluate_models_arrays(
            x,
            y,
            folding_seed=int(folding_seed),
            modeling_seed=int(modeling_seed),
            settings=settings,
        )
        row_scores["LogisticRegression_R2"][seed_number] = result.logistic.r2
        row_scores["LogisticRegression_IMV"][seed_number] = result.logistic.imv
        row_scores["SGD_R2"][seed_number] = result.sgd.r2
        row_scores["SGD_IMV"][seed_number] = result.sgd.imv
        issues += _issue_vector(result)
    return bootstrap_number, row_scores, issues


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npy")
    np.save(temporary, values)
    temporary.replace(path)


def run_crossed_grid(
    raw_data: pd.DataFrame,
    bootstrap_indices: np.ndarray,
    seed_plan: SeedPlan,
    settings: Settings,
    *,
    checkpoint_dir: Path,
    logger: Any,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run the complete shared-bootstrap by paired-seed grid in batches."""

    n_bootstrap = settings.n_bootstrap_resamples
    n_seeds = settings.n_seed_vectors
    matrices = {
        key: np.full((n_bootstrap, n_seeds), np.nan, dtype=float)
        for key in SCORE_KEYS
    }
    issue_counts = np.zeros(len(ISSUE_NAMES), dtype=np.int64)
    folding = seed_plan.folding[:n_seeds]
    modeling = seed_plan.modeling[:n_seeds]
    batch_size = resolve_batch_size(
        n_bootstrap,
        batch_size=settings.bootstrap_batch_size,
        n_jobs=settings.n_jobs,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        with Parallel(n_jobs=settings.n_jobs) as parallel:
            for start in range(0, n_bootstrap, batch_size):
                stop = min(start + batch_size, n_bootstrap)
                results = parallel(
                    delayed(_evaluate_bootstrap_row)(
                        number,
                        bootstrap_indices[number],
                        raw_data,
                        folding,
                        modeling,
                        settings,
                    )
                    for number in range(start, stop)
                )
                for number, row_scores, issues in results:
                    for key in SCORE_KEYS:
                        matrices[key][number] = row_scores[key]
                    issue_counts += issues
                for key, values in matrices.items():
                    _atomic_save_array(
                        checkpoint_dir / f"{SCORE_FILE_STEMS[key]}_checkpoint.npy",
                        values,
                    )
                logger.info(
                    "Crossed grid: %d/%d cells complete (%.1f seconds elapsed)",
                    stop * n_seeds,
                    n_bootstrap * n_seeds,
                    time.monotonic() - started,
                )
    for key, values in matrices.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"crossed experiment has incomplete {key} scores")
    return matrices, issue_counts


def _evaluate_observed_logistic_chunk(
    start: int,
    stop: int,
    features: np.ndarray,
    target: np.ndarray,
    folding_seeds: np.ndarray,
    settings: Settings,
) -> tuple[int, np.ndarray, np.ndarray]:
    values = np.empty((stop - start, 2), dtype=float)
    issues = np.zeros(len(ISSUE_NAMES), dtype=np.int64)
    for local, seed_number in enumerate(range(start, stop)):
        result = _evaluate_logistic_arrays(
            features,
            target,
            folding_seed=int(folding_seeds[seed_number]),
            settings=settings,
        )
        values[local] = (result.r2, result.imv)
        issues += _single_issue_vector(result, model="logistic")
    return start, values, issues


def _evaluate_observed_sgd_chunk(
    start: int,
    stop: int,
    features: np.ndarray,
    target: np.ndarray,
    folding_seeds: np.ndarray,
    modeling_seeds: np.ndarray,
    settings: Settings,
) -> tuple[int, np.ndarray, np.ndarray]:
    values = np.empty((stop - start, 2), dtype=float)
    issues = np.zeros(len(ISSUE_NAMES), dtype=np.int64)
    folding_seed = int(folding_seeds[start])
    if not np.all(folding_seeds[start:stop] == folding_seed):
        raise ValueError("an observed SGD task must contain one folding seed")
    splits = _make_splits(
        features, target, folding_seed=folding_seed, settings=settings
    )
    for local, seed_number in enumerate(range(start, stop)):
        result = _evaluate_model_on_splits(
            features,
            target,
            splits,
            model="sgd",
            modeling_seed=int(modeling_seeds[seed_number]),
            settings=settings,
        )
        values[local] = (result.r2, result.imv)
        issues += _single_issue_vector(result, model="sgd")
    return start, values, issues


def run_observed_seed_sweep(
    features: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    seed_plan: SeedPlan,
    settings: Settings,
    *,
    logger: Any,
) -> ObservedSweep:
    """Run no-bootstrap logistic and factorial SGD sweeps for plotting."""

    x, y = _as_arrays(features, target)
    n_runs = settings.n_visualization_runs
    logistic_chunks = parallel_chunk_ranges(n_runs, n_jobs=settings.n_jobs)
    sgd_folding, sgd_modeling, n_folding, n_modeling = observed_sgd_seed_grid(
        seed_plan, n_runs
    )
    sgd_chunks = [
        (start, start + n_modeling)
        for start in range(0, n_runs, n_modeling)
    ]
    started = time.monotonic()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        with Parallel(n_jobs=settings.n_jobs) as parallel:
            logistic_results = parallel(
                delayed(_evaluate_observed_logistic_chunk)(
                    start, stop, x, y, seed_plan.folding, settings
                )
                for start, stop in logistic_chunks
            )
            sgd_results = parallel(
                delayed(_evaluate_observed_sgd_chunk)(
                    start, stop, x, y, sgd_folding, sgd_modeling, settings
                )
                for start, stop in sgd_chunks
            )
    logistic = np.empty((n_runs, 2), dtype=float)
    sgd = np.empty((n_runs, 2), dtype=float)
    issues = np.zeros(len(ISSUE_NAMES), dtype=np.int64)
    for start, values, counts in logistic_results:
        logistic[start : start + len(values)] = values
        issues += counts
    for start, values, counts in sgd_results:
        sgd[start : start + len(values)] = values
        issues += counts
    if not np.isfinite(logistic).all() or not np.isfinite(sgd).all():
        raise RuntimeError("observed-data sweep finished with incomplete scores")
    logger.info(
        "Observed-data sweeps complete: %d logistic runs and %d SGD runs "
        "(%d folding x %d modeling seeds; %.1f seconds elapsed)",
        n_runs,
        n_runs,
        n_folding,
        n_modeling,
        time.monotonic() - started,
    )
    return ObservedSweep(
        logistic_r2=logistic[:, 0],
        logistic_imv=logistic[:, 1],
        sgd_r2=sgd[:, 0],
        sgd_imv=sgd[:, 1],
        sgd_folding=sgd_folding,
        sgd_modeling=sgd_modeling,
        issue_counts=issues,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _issue_mapping(counts: np.ndarray) -> dict[str, int]:
    return {
        name: int(value) for name, value in zip(ISSUE_NAMES, counts, strict=True)
    }


def write_outputs(
    *,
    output_dir: Path,
    bootstrap_indices: np.ndarray,
    seed_plan: SeedPlan,
    crossed_scores: dict[str, np.ndarray],
    crossed_issue_counts: np.ndarray,
    observed: ObservedSweep,
    diagnostics: dict[str, S5Diagnostics],
    metadata: dict[str, Any],
) -> None:
    """Write run-level scores, seed plans, arrays, and diagnostics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    n_bootstrap, n_seeds = crossed_scores[SCORE_KEYS[0]].shape
    bootstrap_number = np.repeat(np.arange(n_bootstrap), n_seeds)
    seed_number = np.tile(np.arange(n_seeds), n_bootstrap)
    pd.DataFrame(
        {
            "Bootstrap_Replicate": bootstrap_number,
            "Seed_Vector": seed_number,
            "Bootstrap_Seed": seed_plan.bootstrap[bootstrap_number],
            "Folding_Seed": seed_plan.folding[seed_number],
            "Modeling_Seed": seed_plan.modeling[seed_number],
            "LR_R2": crossed_scores["LogisticRegression_R2"].ravel(),
            "LR_IMV": crossed_scores["LogisticRegression_IMV"].ravel(),
            "SGD_R2": crossed_scores["SGD_R2"].ravel(),
            "SGD_IMV": crossed_scores["SGD_IMV"].ravel(),
        }
    ).to_csv(output_dir / "titanic_crossed_bootstrap_scores.csv", index=False)

    n_visual = len(observed.logistic_r2)
    vector_number = np.arange(n_visual)
    logistic_frame = pd.DataFrame(
        {
            "Seed_Vector": vector_number,
            "Folding_Seed": seed_plan.folding[:n_visual],
            "R2": observed.logistic_r2,
            "IMV": observed.logistic_imv,
        }
    )
    sgd_frame = pd.DataFrame(
        {
            "Seed_Vector": vector_number,
            "Folding_Seed": observed.sgd_folding,
            "Modeling_Seed": observed.sgd_modeling,
            "R2": observed.sgd_r2,
            "IMV": observed.sgd_imv,
        }
    )
    logistic_path = output_dir / "titanic_outputs_logistic.csv"
    sgd_path = output_dir / "titanic_outputs_sgd.csv"
    logistic_frame.to_csv(logistic_path, index=False)
    sgd_frame.to_csv(sgd_path, index=False)
    shutil.copyfile(
        logistic_path, output_dir / "titanic_visualization_logistic.csv"
    )
    shutil.copyfile(sgd_path, output_dir / "titanic_visualization_sgd.csv")

    pd.DataFrame(
        {
            "Seed_Vector": np.arange(n_seeds),
            "Folding_Seed": seed_plan.folding[:n_seeds],
            "Modeling_Seed": seed_plan.modeling[:n_seeds],
        }
    ).to_csv(output_dir / "titanic_s5_seed_plan.csv", index=False)
    logistic_frame[["Seed_Vector", "Folding_Seed"]].to_csv(
        output_dir / "titanic_visualization_logistic_seed_plan.csv", index=False
    )
    sgd_frame[["Seed_Vector", "Folding_Seed", "Modeling_Seed"]].to_csv(
        output_dir / "titanic_visualization_sgd_seed_plan.csv", index=False
    )
    pd.DataFrame(
        {
            "Bootstrap_Replicate": np.arange(n_bootstrap),
            "Bootstrap_Seed": seed_plan.bootstrap,
        }
    ).to_csv(output_dir / "titanic_bootstrap_plan.csv", index=False)

    np.save(output_dir / "titanic_bootstrap_indices.npy", bootstrap_indices)
    for key, values in crossed_scores.items():
        np.save(output_dir / f"{SCORE_FILE_STEMS[key]}.npy", values)
    np.save(output_dir / "titanic_crossed_issue_counts.npy", crossed_issue_counts)

    diagnostic_rows = []
    for estimand, result in diagnostics.items():
        model, metric = estimand.rsplit("_", maxsplit=1)
        diagnostic_rows.append(
            {"Estimand": estimand, "Model": model, "Metric": metric, **result.to_dict()}
        )
    pd.DataFrame(diagnostic_rows).to_csv(
        output_dir / "titanic_s5_diagnostics.csv", index=False
    )
    payload = {
        "metadata": metadata,
        "diagnostics": {
            estimand: result.to_dict() for estimand, result in diagnostics.items()
        },
        "crossed_issue_counts": _issue_mapping(crossed_issue_counts),
        "observed_issue_counts": _issue_mapping(observed.issue_counts),
        "observed_data_seed_summaries": {
            "LogisticRegression_R2": observed_seed_summary(observed.logistic_r2).to_dict(),
            "LogisticRegression_IMV": observed_seed_summary(observed.logistic_imv).to_dict(),
            "SGD_R2": observed_seed_summary(observed.sgd_r2).to_dict(),
            "SGD_IMV": observed_seed_summary(observed.sgd_imv).to_dict(),
        },
    }
    (output_dir / "titanic_s5_diagnostics.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_fixed_seed_references(
    accuracy_path: Path,
    features: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    settings: Settings,
) -> np.ndarray:
    """Preserve figure annotations for canonical seeds 42 and 123."""

    lines: list[str] = []
    issues = np.zeros(len(ISSUE_NAMES), dtype=np.int64)
    for seed in (42, 123):
        result = evaluate_models(
            features,
            target,
            folding_seed=seed,
            modeling_seed=seed,
            settings=settings,
        )
        issues += _issue_vector(result)
        lines.append(
            f"seed={seed}, sgd_accuracy={result.sgd.accuracy:.4f}, "
            f"lr_accuracy={result.logistic.accuracy:.4f}, "
            f"sgd_imv={result.sgd.imv:.4f}, lr_imv={result.logistic.imv:.4f}"
        )
    accuracy_path.parent.mkdir(parents=True, exist_ok=True)
    accuracy_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--titanic-csv", type=Path, default=None)
    parser.add_argument("--seed-list", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--accuracy-path", type=Path, default=None)
    parser.add_argument("--n-bootstrap-resamples", type=int, default=100)
    parser.add_argument("--n-seed-vectors", type=int, default=100)
    parser.add_argument("--n-visualization-runs", type=int, default=1_000)
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="parallel workers; -1 uses all available CPUs (default: -1)",
    )
    parser.add_argument(
        "--bootstrap-batch-size", type=int, default=0,
        help="bootstrap rows per checkpoint batch; 0 uses one worker wave",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--sgd-max-iter", type=int, default=1_000)
    parser.add_argument("--sgd-tolerance", type=float, default=1e-4)
    parser.add_argument("--logistic-max-iter", type=int, default=250)
    parser.add_argument(
        "--reuse-crossed-scores",
        action="store_true",
        help=(
            "load completed crossed arrays from --output-dir; shapes and finite "
            "values are validated before reporting"
        ),
    )
    return parser


def _load_reused_crossed_scores(
    output_dir: Path, settings: Settings
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    expected_shape = (settings.n_bootstrap_resamples, settings.n_seed_vectors)
    scores: dict[str, np.ndarray] = {}
    for key in SCORE_KEYS:
        values = np.load(output_dir / f"{SCORE_FILE_STEMS[key]}.npy")
        if values.shape != expected_shape:
            raise ValueError(
                f"saved {key} array has shape {values.shape}; expected {expected_shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"saved {key} array must be complete and finite")
        scores[key] = values
    issue_path = output_dir / "titanic_crossed_issue_counts.npy"
    if issue_path.exists():
        issue_counts = np.load(issue_path)
        if issue_counts.shape != (len(ISSUE_NAMES),):
            raise ValueError("saved crossed issue counts have the wrong shape")
    else:
        issue_counts = np.full(len(ISSUE_NAMES), -1, dtype=np.int64)
    return scores, issue_counts


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = Settings(
        n_bootstrap_resamples=args.n_bootstrap_resamples,
        n_seed_vectors=args.n_seed_vectors,
        n_visualization_runs=args.n_visualization_runs,
        n_jobs=args.n_jobs,
        bootstrap_batch_size=args.bootstrap_batch_size,
        n_folds=args.n_folds,
        sgd_max_iter=args.sgd_max_iter,
        sgd_tolerance=args.sgd_tolerance,
        logistic_max_iter=args.logistic_max_iter,
    )
    settings.validate()

    repo_root = Path(__file__).resolve().parents[2]
    titanic_csv = args.titanic_csv or repo_root / "data/titanic/raw/train.csv"
    seed_path = args.seed_list or repo_root / "assets/seed_list.txt"
    output_dir = args.output_dir or repo_root / "data/titanic/results"
    log_path = args.log_path or repo_root / "results/results_logs/titanic_s5_diagnostics.log"
    accuracy_path = args.accuracy_path or repo_root / "data/titanic/accuracy_seeds.txt"
    logger = configure_run_logger("titanic_s5", log_path)
    logger.info("Starting Titanic S5 analysis")
    logger.info("Titanic data: %s", titanic_csv)
    logger.info("Seed list: %s", seed_path)
    logger.info("Run-level output directory: %s", output_dir)
    logger.info(
        "Parallel workers: %d (n_jobs=%d); bootstrap batch size: %d",
        resolve_worker_count(settings.n_jobs),
        settings.n_jobs,
        resolve_batch_size(
            settings.n_bootstrap_resamples,
            batch_size=settings.bootstrap_batch_size,
            n_jobs=settings.n_jobs,
        ),
    )
    started = time.monotonic()

    raw_data = read_titanic_data(titanic_csv)
    observed_features, observed_target = wrangle_titanic(raw_data)
    seeds = load_seed_list(seed_path)
    seed_plan = make_seed_plan(seeds, settings)
    bootstrap_indices = bootstrap_index_block(
        len(raw_data), [int(seed) for seed in seed_plan.bootstrap]
    )
    logger.info(
        "Prepared %d shared raw-row bootstrap datasets crossed with %d paired "
        "seed vectors (%d cells; both models and both scores per cell)",
        settings.n_bootstrap_resamples,
        settings.n_seed_vectors,
        settings.n_bootstrap_resamples * settings.n_seed_vectors,
    )

    if args.reuse_crossed_scores:
        crossed_scores, crossed_issues = _load_reused_crossed_scores(output_dir, settings)
        logger.info(
            "Reused and validated %d crossed cells for each of four estimands",
            settings.n_bootstrap_resamples * settings.n_seed_vectors,
        )
        if np.any(crossed_issues < 0):
            logger.warning(
                "Crossed issue-count artifact was absent; reused-run counts are unknown"
            )
    else:
        crossed_scores, crossed_issues = run_crossed_grid(
            raw_data,
            bootstrap_indices,
            seed_plan,
            settings,
            checkpoint_dir=output_dir,
            logger=logger,
        )

    observed = run_observed_seed_sweep(
        observed_features, observed_target, seed_plan, settings, logger=logger
    )
    fixed_seed_issues = write_fixed_seed_references(
        accuracy_path, observed_features, observed_target, settings
    )
    diagnostics = {
        key: crossed_s5_diagnostics(crossed_scores[key]) for key in SCORE_KEYS
    }
    metadata = {
        "analysis": "Titanic",
        "settings": asdict(settings),
        "resolved_parallel_workers": resolve_worker_count(settings.n_jobs),
        "input_csv": str(titanic_csv.resolve()),
        "input_sha256": _sha256(titanic_csv),
        "n_raw_observations": len(raw_data),
        "n_wrangled_predictors": observed_features.shape[1],
        "seed_list": str(seed_path.resolve()),
        "seed_allocation": "stage-major disjoint blocks: folding, SGD modeling, then bootstrap",
        "primary_visualization_metric": "IMV",
        "IMV_entropy_solver": (
            "bounded Brent root on the upper binary-entropy branch; "
            "LRU-cached by likelihood"
        ),
        "crossed_issue_counts": _issue_mapping(crossed_issues),
        "observed_issue_counts": _issue_mapping(observed.issue_counts),
        "fixed_seed_issue_counts": _issue_mapping(fixed_seed_issues),
        "versions": {
            "Python": platform.python_version(),
            "NumPy": np.__version__,
            "pandas": pd.__version__,
            "SciPy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
            "joblib": joblib.__version__,
        },
    }
    write_outputs(
        output_dir=output_dir,
        bootstrap_indices=bootstrap_indices,
        seed_plan=seed_plan,
        crossed_scores=crossed_scores,
        crossed_issue_counts=crossed_issues,
        observed=observed,
        diagnostics=diagnostics,
        metadata=metadata,
    )

    common_details = {
        "within_cell_replications": (
            f"1 complete {settings.n_folds}-fold evaluation; cell score is fold mean"
        ),
        "bootstrap_design": (
            "crossed; each raw-row resample is wrangled once and reused for every "
            "paired seed vector and both models"
        ),
        "bootstrap_PRNG": f"NumPy PCG64 {np.__version__}",
        "folding_PRNG": (
            "integer random_state via scikit-learn KFold/NumPy RandomState "
            f"(MT19937); scikit-learn {sklearn.__version__}"
        ),
        "evaluation": (
            f"{settings.n_folds}-fold shuffled KFold (unstratified), with the same "
            "folds supplied to deterministic Logit and seeded SGD"
        ),
        "preprocessing": (
            "legacy feature construction before KFold within each D*: age sex/class "
            "medians and embarked mode are re-estimated per bootstrap"
        ),
        "joint_or_stagewise_variation": (
            "joint paired vector for SGD (folding + modeling); Logit varies folding only"
        ),
        "fallback_policy": (
            "legacy training-prevalence prediction on fit failure and zero on "
            "nonfinite score; every occurrence is counted below"
        ),
        "IMV_entropy_solver": (
            "bounded Brent root on the upper binary-entropy branch; cached by "
            "likelihood (mathematically equivalent to the former scalar minimization)"
        ),
        "issue_count_unit": "fold-level events summed across all crossed cells",
        "parallel_backend": "joblib loky processes; one inner numerical thread",
        "parallel_workers": resolve_worker_count(settings.n_jobs),
        "bootstrap_checkpoint_batch_size": resolve_batch_size(
            settings.n_bootstrap_resamples,
            batch_size=settings.bootstrap_batch_size,
            n_jobs=settings.n_jobs,
        ),
        **{
            f"crossed_{key}": value
            for key, value in _issue_mapping(crossed_issues).items()
        },
    }
    model_details = {
        "LogisticRegression": {
            "model": (
                "statsmodels Logit with intercept; deterministic conditional on "
                f"data/fold; maxiter={settings.logistic_max_iter}"
            ),
            "algorithmic_seed_components": "folding/evaluation seed",
        },
        "SGD": {
            "model": (
                "sklearn SGDClassifier(loss=log_loss, "
                f"max_iter={settings.sgd_max_iter}, tol={settings.sgd_tolerance})"
            ),
            "algorithmic_seed_components": "folding/evaluation and SGD modeling seeds",
            "modeling_PRNG": (
                "integer random_state via scikit-learn/NumPy RandomState "
                f"(MT19937); scikit-learn {sklearn.__version__}; NumPy {np.__version__}"
            ),
        },
    }
    for estimand, result in diagnostics.items():
        model, metric = estimand.rsplit("_", maxsplit=1)
        details = {
            **common_details,
            **model_details[model],
            "score": (
                "legacy probability pseudo-R2"
                if metric == "R2"
                else "legacy likelihood-to-entropy IMV"
            ),
        }
        logger.info(
            "\n%s",
            format_s5_report(
                result, estimand=estimand, computational_details=details
            ),
        )

    observed_scores = {
        "LogisticRegression_R2": observed.logistic_r2,
        "LogisticRegression_IMV": observed.logistic_imv,
        "SGD_R2": observed.sgd_r2,
        "SGD_IMV": observed.sgd_imv,
    }
    for estimand, values in observed_scores.items():
        logger.info(
            "\n%s",
            format_observed_seed_report(
                observed_seed_summary(values), estimand=estimand
            ),
        )
    logger.info("Crossed issue counts: %s", _issue_mapping(crossed_issues))
    logger.info("Observed issue counts: %s", _issue_mapping(observed.issue_counts))
    logger.info("Fixed-seed issue counts: %s", _issue_mapping(fixed_seed_issues))
    substitution_positions = np.array([0, 2, 3, 5])
    if np.any(crossed_issues[substitution_positions] > 0):
        logger.warning(
            "The crossed grid used one or more fit fallbacks/metric substitutions; "
            "interpret the S5 estimates with those disclosed events in mind"
        )
    if np.any(crossed_issues[[1, 4]] > 0):
        logger.warning(
            "The crossed grid emitted model-fit warnings; counts are reported above"
        )
    validate_s5_log(log_path, tuple(diagnostics))
    logger.info("Wrote fixed-seed figure references: %s", accuracy_path)
    logger.info("Titanic analysis complete in %.1f seconds", time.monotonic() - started)


if __name__ == "__main__":
    main()
