"""Run the California-housing seed and sampling-uncertainty analysis.

The diagnostic experiment follows Supplementary Sections S4.2.1 and S5:
one shared block of bootstrap datasets is crossed with a set of algorithmic
seed vectors.  A separate, no-bootstrap seed sweep supplies the distributions
used by the visualisations; it is never mixed into the S5 variance estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split

try:  # Works both as a package import and as a directly executed script.
    from .reporting_utils import (
        S5Diagnostics,
        bootstrap_index_block,
        configure_run_logger,
        crossed_s5_diagnostics,
        format_observed_seed_report,
        format_s5_report,
        load_seed_list,
        observed_seed_summary,
        seed_component_blocks,
    )
except ImportError:  # pragma: no cover - exercised by direct CLI execution.
    from reporting_utils import (  # type: ignore[no-redef]
        S5Diagnostics,
        bootstrap_index_block,
        configure_run_logger,
        crossed_s5_diagnostics,
        format_observed_seed_report,
        format_s5_report,
        load_seed_list,
        observed_seed_summary,
        seed_component_blocks,
    )


TARGET = "median_house_value"
OCEAN_CODES = {
    "<1H OCEAN": 0,
    "INLAND": 1,
    "NEAR OCEAN": 2,
    "NEAR BAY": 3,
    "ISLAND": 4,
}
SCORE_NAME = "explained_variance_score (legacy output column: R2)"


@dataclass(frozen=True)
class Settings:
    """Controls for both the S5 grid and observed-data seed sweep."""

    n_bootstrap_resamples: int = 100
    n_seed_vectors: int = 100
    n_visualization_runs: int = 1_000
    n_jobs: int = 10
    bootstrap_batch_size: int = 10
    test_size: float = 0.30
    n_estimators: int = 25
    max_depth: int = 5

    def validate(self) -> None:
        if self.n_bootstrap_resamples < 2 or self.n_seed_vectors < 2:
            raise ValueError("S5 diagnostics require at least two bootstraps and seeds")
        if self.n_visualization_runs < 2:
            raise ValueError("n_visualization_runs must be at least two")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")
        if self.bootstrap_batch_size < 1:
            raise ValueError("bootstrap_batch_size must be positive")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must lie strictly between zero and one")
        if self.n_estimators < 1 or self.max_depth < 1:
            raise ValueError("random-forest size settings must be positive")


@dataclass(frozen=True)
class SeedPlan:
    """Stage-specific algorithm seeds and independent bootstrap seeds."""

    folding: np.ndarray
    modeling: np.ndarray
    bootstrap: np.ndarray


@dataclass(frozen=True)
class ScorePair:
    """OLS and random-forest scores from the same split."""

    ols: float
    random_forest: float


def read_housing_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load complete cases and deterministically encode ocean proximity."""

    data = pd.read_csv(csv_path)
    if TARGET not in data.columns:
        raise KeyError(f"housing data must contain {TARGET!r}")
    if "ocean_proximity" in data.columns:
        ocean = data["ocean_proximity"]
        encoded = ocean.map(OCEAN_CODES)
        unknown_mask = encoded.isna() & ocean.notna()
        if unknown_mask.any():
            unknown = sorted(
                data.loc[unknown_mask, "ocean_proximity"].dropna().unique()
            )
            raise ValueError(f"unknown ocean_proximity values: {unknown}")
        data["ocean_proximity"] = encoded

    data = data.dropna().reset_index(drop=True)
    features = data.drop(columns=TARGET)
    non_numeric = features.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise TypeError(f"all predictors must be numeric; found {non_numeric}")
    return (
        features.to_numpy(dtype=np.float64, copy=True),
        data[TARGET].to_numpy(dtype=np.float64, copy=True),
    )


def make_seed_plan(seed_list: list[int], settings: Settings) -> SeedPlan:
    """Create independent stage blocks and align them as joint seed vectors."""

    n_vectors = max(settings.n_seed_vectors, settings.n_visualization_runs)
    components = seed_component_blocks(
        seed_list,
        n_vectors=n_vectors,
        component_names=("folding", "modeling"),
    )
    bootstrap_start = 2 * n_vectors
    bootstrap_stop = bootstrap_start + settings.n_bootstrap_resamples
    if bootstrap_stop > len(seed_list):
        raise ValueError(
            f"seed list has {len(seed_list)} entries but {bootstrap_stop} are needed"
        )
    bootstrap_seeds = np.asarray(
        seed_list[bootstrap_start:bootstrap_stop], dtype=np.uint64
    )
    return SeedPlan(
        folding=components["folding"],
        modeling=components["modeling"],
        bootstrap=bootstrap_seeds,
    )


def _split_dataset(
    features: np.ndarray,
    target: np.ndarray,
    *,
    folding_seed: int,
    settings: Settings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        features,
        target,
        test_size=settings.test_size,
        random_state=int(folding_seed),
        shuffle=True,
    )


def _ols_score(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> float:
    ols = LinearRegression()
    ols.fit(x_train, y_train)
    return float(explained_variance_score(y_test, ols.predict(x_test)))


def _random_forest_score(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    modeling_seed: int,
    settings: Settings,
) -> float:
    forest = RandomForestRegressor(
        n_estimators=settings.n_estimators,
        max_depth=settings.max_depth,
        random_state=int(modeling_seed),
        bootstrap=True,
        n_jobs=1,
    )
    forest.fit(x_train, y_train)
    return float(explained_variance_score(y_test, forest.predict(x_test)))


def evaluate_models(
    features: np.ndarray,
    target: np.ndarray,
    *,
    folding_seed: int,
    modeling_seed: int,
    settings: Settings,
) -> ScorePair:
    """Evaluate OLS and RF on one seeded 70/30 split of one dataset."""

    split = _split_dataset(
        features, target, folding_seed=folding_seed, settings=settings
    )
    return ScorePair(
        ols=_ols_score(*split),
        random_forest=_random_forest_score(
            *split, modeling_seed=modeling_seed, settings=settings
        ),
    )


def evaluate_ols(
    features: np.ndarray,
    target: np.ndarray,
    *,
    folding_seed: int,
    settings: Settings,
) -> float:
    """Evaluate deterministic OLS on one seeded observed-data split."""

    split = _split_dataset(
        features, target, folding_seed=folding_seed, settings=settings
    )
    return _ols_score(*split)


def evaluate_random_forest(
    features: np.ndarray,
    target: np.ndarray,
    *,
    folding_seed: int,
    modeling_seed: int,
    settings: Settings,
) -> float:
    """Evaluate RF on one seeded observed-data split and model fit."""

    split = _split_dataset(
        features, target, folding_seed=folding_seed, settings=settings
    )
    return _random_forest_score(
        *split, modeling_seed=modeling_seed, settings=settings
    )


def _evaluate_bootstrap_row(
    bootstrap_index: int,
    row_indices: np.ndarray,
    features: np.ndarray,
    target: np.ndarray,
    folding_seeds: np.ndarray,
    modeling_seeds: np.ndarray,
    settings: Settings,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Evaluate every seed vector on one shared bootstrap dataset."""

    sampled_features = features[row_indices]
    sampled_target = target[row_indices]
    ols_scores = np.empty(len(folding_seeds), dtype=float)
    rf_scores = np.empty(len(folding_seeds), dtype=float)
    for seed_index, (folding_seed, modeling_seed) in enumerate(
        zip(folding_seeds, modeling_seeds, strict=True)
    ):
        scores = evaluate_models(
            sampled_features,
            sampled_target,
            folding_seed=int(folding_seed),
            modeling_seed=int(modeling_seed),
            settings=settings,
        )
        ols_scores[seed_index] = scores.ols
        rf_scores[seed_index] = scores.random_forest
    return bootstrap_index, ols_scores, rf_scores


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    """Checkpoint an array without exposing a partially written final file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npy")
    np.save(temporary, values)
    temporary.replace(path)


def run_crossed_grid(
    features: np.ndarray,
    target: np.ndarray,
    bootstrap_indices: np.ndarray,
    seed_plan: SeedPlan,
    settings: Settings,
    *,
    checkpoint_dir: Path,
    logger: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the complete shared-bootstrap by seed-vector grid in batches."""

    n_bootstrap = settings.n_bootstrap_resamples
    n_seeds = settings.n_seed_vectors
    ols_matrix = np.full((n_bootstrap, n_seeds), np.nan, dtype=float)
    rf_matrix = np.full((n_bootstrap, n_seeds), np.nan, dtype=float)
    folding_seeds = seed_plan.folding[:n_seeds]
    modeling_seeds = seed_plan.modeling[:n_seeds]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    with Parallel(n_jobs=settings.n_jobs, prefer="processes") as parallel:
        for start in range(0, n_bootstrap, settings.bootstrap_batch_size):
            stop = min(start + settings.bootstrap_batch_size, n_bootstrap)
            batch_results = parallel(
                delayed(_evaluate_bootstrap_row)(
                    bootstrap_index,
                    bootstrap_indices[bootstrap_index],
                    features,
                    target,
                    folding_seeds,
                    modeling_seeds,
                    settings,
                )
                for bootstrap_index in range(start, stop)
            )
            for bootstrap_index, ols_scores, rf_scores in batch_results:
                ols_matrix[bootstrap_index] = ols_scores
                rf_matrix[bootstrap_index] = rf_scores

            _atomic_save_array(
                checkpoint_dir / "housing_crossed_ols_checkpoint.npy", ols_matrix
            )
            _atomic_save_array(
                checkpoint_dir / "housing_crossed_rf_checkpoint.npy", rf_matrix
            )
            completed = stop * n_seeds
            logger.info(
                "Crossed grid: %d/%d cells complete (%.1f seconds elapsed)",
                completed,
                n_bootstrap * n_seeds,
                time.monotonic() - started,
            )

    if not np.isfinite(ols_matrix).all() or not np.isfinite(rf_matrix).all():
        raise RuntimeError("crossed experiment finished with incomplete scores")
    return ols_matrix, rf_matrix


def _evaluate_observed_ols_chunk(
    start: int,
    stop: int,
    features: np.ndarray,
    target: np.ndarray,
    folding_seeds: np.ndarray,
    settings: Settings,
) -> tuple[int, np.ndarray]:
    ols_scores = np.empty(stop - start, dtype=float)
    for local_index, seed_index in enumerate(range(start, stop)):
        ols_scores[local_index] = evaluate_ols(
            features,
            target,
            folding_seed=int(folding_seeds[seed_index]),
            settings=settings,
        )
    return start, ols_scores


def _evaluate_observed_rf_chunk(
    start: int,
    stop: int,
    features: np.ndarray,
    target: np.ndarray,
    folding_seeds: np.ndarray,
    modeling_seeds: np.ndarray,
    settings: Settings,
) -> tuple[int, np.ndarray]:
    rf_scores = np.empty(stop - start, dtype=float)
    for local_index, seed_index in enumerate(range(start, stop)):
        rf_scores[local_index] = evaluate_random_forest(
            features,
            target,
            folding_seed=int(folding_seeds[seed_index]),
            modeling_seed=int(modeling_seeds[seed_index]),
            settings=settings,
        )
    return start, rf_scores


def observed_rf_seed_grid(
    seed_plan: SeedPlan, n_runs: int
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Make a near-square full factorial with exactly ``n_runs`` pairs."""

    n_modeling = math.isqrt(n_runs)
    while n_runs % n_modeling:
        n_modeling -= 1
    n_folding = n_runs // n_modeling
    folding = np.repeat(seed_plan.folding[:n_folding], n_modeling)
    modeling = np.tile(seed_plan.modeling[:n_modeling], n_folding)
    return folding, modeling, n_folding, n_modeling


def run_observed_seed_sweep(
    features: np.ndarray,
    target: np.ndarray,
    seed_plan: SeedPlan,
    settings: Settings,
    *,
    logger: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run no-bootstrap OLS and factorial RF sweeps for visualization."""

    n_runs = settings.n_visualization_runs
    chunk_size = max(1, min(50, n_runs))
    chunks = [
        (start, min(start + chunk_size, n_runs))
        for start in range(0, n_runs, chunk_size)
    ]
    rf_folding, rf_modeling, n_folding, n_modeling = observed_rf_seed_grid(
        seed_plan, n_runs
    )
    started = time.monotonic()
    with Parallel(n_jobs=settings.n_jobs, prefer="processes") as parallel:
        ols_results = parallel(
            delayed(_evaluate_observed_ols_chunk)(
                start,
                stop,
                features,
                target,
                seed_plan.folding,
                settings,
            )
            for start, stop in chunks
        )
        rf_results = parallel(
            delayed(_evaluate_observed_rf_chunk)(
                start,
                stop,
                features,
                target,
                rf_folding,
                rf_modeling,
                settings,
            )
            for start, stop in chunks
        )
    ols_scores = np.empty(n_runs, dtype=float)
    rf_scores = np.empty(n_runs, dtype=float)
    for start, ols_chunk in ols_results:
        stop = start + len(ols_chunk)
        ols_scores[start:stop] = ols_chunk
    for start, rf_chunk in rf_results:
        stop = start + len(rf_chunk)
        rf_scores[start:stop] = rf_chunk
    logger.info(
        "Observed-data sweeps complete: %d OLS runs and %d RF runs "
        "(%d folding x %d modeling seeds; %.1f seconds elapsed)",
        n_runs,
        n_runs,
        n_folding,
        n_modeling,
        time.monotonic() - started,
    )
    return ols_scores, rf_scores, rf_folding, rf_modeling


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict JSON values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_outputs(
    *,
    output_dir: Path,
    bootstrap_indices: np.ndarray,
    seed_plan: SeedPlan,
    crossed_ols: np.ndarray,
    crossed_rf: np.ndarray,
    observed_ols: np.ndarray,
    observed_rf: np.ndarray,
    observed_rf_folding_seeds: np.ndarray,
    observed_rf_modeling_seeds: np.ndarray,
    diagnostics: dict[str, S5Diagnostics],
    metadata: dict[str, Any],
) -> None:
    """Write run-level outputs, seed plans, and machine-readable diagnostics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    n_bootstrap, n_seeds = crossed_rf.shape
    bootstrap_number = np.repeat(np.arange(n_bootstrap), n_seeds)
    seed_number = np.tile(np.arange(n_seeds), n_bootstrap)

    pd.DataFrame(
        {
            "Bootstrap_Replicate": bootstrap_number,
            "Seed_Vector": seed_number,
            "Bootstrap_Seed": seed_plan.bootstrap[bootstrap_number],
            "Folding_Seed": seed_plan.folding[seed_number],
            "Modeling_Seed": seed_plan.modeling[seed_number],
            "OLS_R2": crossed_ols.ravel(),
            "RF_R2": crossed_rf.ravel(),
        }
    ).to_csv(output_dir / "housing_crossed_bootstrap_scores.csv", index=False)

    n_visual = len(observed_rf)
    vector_number = np.arange(n_visual)
    rf_frame = pd.DataFrame(
        {
            "Seed_Vector": vector_number,
            "Folding_Seed": observed_rf_folding_seeds,
            "Modeling_Seed": observed_rf_modeling_seeds,
            "R2": observed_rf,
        }
    )
    ols_frame = pd.DataFrame(
        {
            "Seed_Vector": vector_number,
            "Folding_Seed": seed_plan.folding[:n_visual],
            "R2": observed_ols,
        }
    )
    rf_frame.to_csv(output_dir / "housing_outputs_rf.csv", index=False)
    rf_frame.to_csv(output_dir / "r2.csv", index=False)
    ols_frame.to_csv(output_dir / "housing_outputs_ols.csv", index=False)
    rf_frame.to_csv(output_dir / "housing_visualization_rf.csv", index=False)
    ols_frame.to_csv(output_dir / "housing_visualization_ols.csv", index=False)

    s5_vector_number = np.arange(n_seeds)
    s5_seed_frame = pd.DataFrame(
        {
            "Seed_Vector": s5_vector_number,
            "Folding_Seed": seed_plan.folding[:n_seeds],
            "Modeling_Seed": seed_plan.modeling[:n_seeds],
        }
    )
    s5_seed_frame.to_csv(output_dir / "housing_s5_seed_plan.csv", index=False)
    s5_seed_frame.to_csv(output_dir / "housing_seed_plan.csv", index=False)
    ols_frame[["Seed_Vector", "Folding_Seed"]].to_csv(
        output_dir / "housing_visualization_ols_seed_plan.csv", index=False
    )
    rf_frame[["Seed_Vector", "Folding_Seed", "Modeling_Seed"]].to_csv(
        output_dir / "housing_visualization_rf_seed_plan.csv", index=False
    )
    pd.DataFrame(
        {
            "Bootstrap_Replicate": np.arange(n_bootstrap),
            "Bootstrap_Seed": seed_plan.bootstrap,
        }
    ).to_csv(output_dir / "housing_bootstrap_plan.csv", index=False)

    np.save(output_dir / "housing_bootstrap_indices.npy", bootstrap_indices)
    np.save(output_dir / "housing_crossed_ols_scores.npy", crossed_ols)
    np.save(output_dir / "housing_crossed_rf_scores.npy", crossed_rf)

    diagnostics_rows = []
    for model, result in diagnostics.items():
        diagnostics_rows.append({"Model": model, **result.to_dict()})
    pd.DataFrame(diagnostics_rows).to_csv(
        output_dir / "housing_s5_diagnostics.csv", index=False
    )
    payload = {
        "metadata": metadata,
        "diagnostics": {
            model: result.to_dict() for model, result in diagnostics.items()
        },
        "observed_data_seed_summaries": {
            "OLS": observed_seed_summary(observed_ols).to_dict(),
            "RandomForest": observed_seed_summary(observed_rf).to_dict(),
        },
    }
    (output_dir / "housing_s5_diagnostics.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_fixed_seed_references(
    housing_dir: Path,
    features: np.ndarray,
    target: np.ndarray,
    settings: Settings,
) -> None:
    """Preserve the figure annotations for canonical seeds 42 and 123."""

    lines = []
    for seed in (42, 123):
        result = evaluate_models(
            features,
            target,
            folding_seed=seed,
            modeling_seed=seed,
            settings=settings,
        )
        lines.append(
            f"seed={seed}, rf_R2={result.random_forest:.4f}, "
            f"ols_R2={result.ols:.4f}"
        )
    (housing_dir / "accuracy_seeds.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--housing-csv", type=Path, default=None)
    parser.add_argument("--seed-list", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--n-bootstrap-resamples", type=int, default=100)
    parser.add_argument("--n-seed-vectors", type=int, default=100)
    parser.add_argument("--n-visualization-runs", type=int, default=1_000)
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--bootstrap-batch-size", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--n-estimators", type=int, default=25)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument(
        "--reuse-crossed-scores",
        action="store_true",
        help=(
            "load already completed crossed score arrays from --output-dir; "
            "their shape and finiteness are validated before reporting"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = Settings(
        n_bootstrap_resamples=args.n_bootstrap_resamples,
        n_seed_vectors=args.n_seed_vectors,
        n_visualization_runs=args.n_visualization_runs,
        n_jobs=args.n_jobs,
        bootstrap_batch_size=args.bootstrap_batch_size,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    settings.validate()

    repo_root = Path(__file__).resolve().parents[2]
    housing_csv = args.housing_csv or repo_root / "data/housing/raw/housing.csv"
    seed_path = args.seed_list or repo_root / "assets/seed_list.txt"
    output_dir = args.output_dir or repo_root / "data/housing/results"
    log_path = (
        args.log_path
        or repo_root / "results/results_logs/housing_s5_diagnostics.log"
    )
    logger = configure_run_logger("housing_s5", log_path)
    logger.info("Starting housing S5 analysis")
    logger.info("Housing data: %s", housing_csv)
    logger.info("Seed list: %s", seed_path)
    logger.info("Run-level output directory: %s", output_dir)

    started = time.monotonic()
    features, target = read_housing_data(housing_csv)
    seeds = load_seed_list(seed_path)
    seed_plan = make_seed_plan(seeds, settings)
    indices = bootstrap_index_block(
        len(target), [int(seed) for seed in seed_plan.bootstrap]
    )
    logger.info(
        "Prepared one shared block of %d bootstrap datasets crossed with %d "
        "seed vectors (%d cells)",
        settings.n_bootstrap_resamples,
        settings.n_seed_vectors,
        settings.n_bootstrap_resamples * settings.n_seed_vectors,
    )

    if args.reuse_crossed_scores:
        crossed_ols = np.load(output_dir / "housing_crossed_ols_scores.npy")
        crossed_rf = np.load(output_dir / "housing_crossed_rf_scores.npy")
        expected_shape = (
            settings.n_bootstrap_resamples,
            settings.n_seed_vectors,
        )
        if crossed_ols.shape != expected_shape or crossed_rf.shape != expected_shape:
            raise ValueError(
                "saved crossed score arrays do not match requested shape "
                f"{expected_shape}: OLS={crossed_ols.shape}, RF={crossed_rf.shape}"
            )
        if not np.isfinite(crossed_ols).all() or not np.isfinite(crossed_rf).all():
            raise ValueError("saved crossed score arrays must be complete and finite")
        logger.info(
            "Reused and validated %d completed crossed cells per model from %s",
            crossed_ols.size,
            output_dir,
        )
    else:
        crossed_ols, crossed_rf = run_crossed_grid(
            features,
            target,
            indices,
            seed_plan,
            settings,
            checkpoint_dir=output_dir,
            logger=logger,
        )
    (
        observed_ols,
        observed_rf,
        observed_rf_folding,
        observed_rf_modeling,
    ) = run_observed_seed_sweep(features, target, seed_plan, settings, logger=logger)

    diagnostics = {
        "OLS": crossed_s5_diagnostics(crossed_ols),
        "RandomForest": crossed_s5_diagnostics(crossed_rf),
    }
    common_details = {
        "within_cell_replications": 1,
        "bootstrap_design": "crossed; same resample block reused for every seed",
        "bootstrap_PRNG": f"NumPy PCG64 {np.__version__}",
        "algorithmic_PRNG": (
            "integer random_state via scikit-learn/NumPy RandomState "
            f"(MT19937); scikit-learn {sklearn.__version__}; NumPy {np.__version__}"
        ),
        "Python_version": platform.python_version(),
        "joblib_version": joblib.__version__,
        "score": SCORE_NAME,
        "test_fraction": settings.test_size,
        "external_dataset_bootstrap": (
            "B shared nonparametric resamples of D_obs, generated once and "
            "crossed with every seed vector"
        ),
    }
    ols_details = {
        **common_details,
        "seed_components": "folding_seed",
        "seed_components_varied": "jointly (one evaluation component)",
        "model": "LinearRegression (deterministic conditional on data and split)",
    }
    rf_details = {
        **common_details,
        "seed_components": "folding_seed, modeling_seed",
        "seed_components_varied": "jointly as paired seed vectors",
        "model": (
            "RandomForestRegressor"
            f"(n_estimators={settings.n_estimators}, max_depth={settings.max_depth})"
        ),
        "random_forest_internal_bootstrap": (
            "True; model-internal tree bootstrapping is distinct from the "
            "external dataset bootstrap"
        ),
    }

    logger.info(
        "\n%s",
        format_s5_report(
            diagnostics["OLS"],
            estimand=f"OLS {SCORE_NAME}",
            computational_details=ols_details,
        ),
    )
    logger.info(
        "\n%s",
        format_s5_report(
            diagnostics["RandomForest"],
            estimand=f"Random forest {SCORE_NAME}",
            computational_details=rf_details,
        ),
    )
    logger.info(
        "Visualization sweep design: OLS uses %d distinct folding seeds; RF "
        "uses %d distinct folding x %d distinct modeling seeds in a full "
        "factorial (%d seed vectors). No external dataset bootstrap is used.",
        settings.n_visualization_runs,
        np.unique(observed_rf_folding).size,
        np.unique(observed_rf_modeling).size,
        settings.n_visualization_runs,
    )
    logger.info(
        "\n%s",
        format_observed_seed_report(
            observed_seed_summary(observed_ols), estimand=f"OLS {SCORE_NAME}"
        ),
    )
    logger.info(
        "\n%s",
        format_observed_seed_report(
            observed_seed_summary(observed_rf),
            estimand=f"Random forest {SCORE_NAME}",
        ),
    )

    metadata = {
        "housing_csv": str(housing_csv),
        "housing_csv_sha256": _sha256(housing_csv),
        "seed_list": str(seed_path),
        "seed_list_sha256": _sha256(seed_path),
        "n_complete_observations": len(target),
        "n_features": features.shape[1],
        "settings": settings.__dict__,
        "crossed_scores_reused_for_this_reporting_pass": bool(
            args.reuse_crossed_scores
        ),
        "OLS_computational_details": ols_details,
        "RandomForest_computational_details": rf_details,
        "visualization_sweeps": {
            "external_dataset_bootstrap": False,
            "OLS": {
                "runs": settings.n_visualization_runs,
                "design": "distinct folding seeds",
            },
            "RandomForest": {
                "runs": settings.n_visualization_runs,
                "design": "full factorial folding_seed x modeling_seed",
                "n_folding_seeds": int(np.unique(observed_rf_folding).size),
                "n_modeling_seeds": int(np.unique(observed_rf_modeling).size),
            },
        },
    }
    write_outputs(
        output_dir=output_dir,
        bootstrap_indices=indices,
        seed_plan=seed_plan,
        crossed_ols=crossed_ols,
        crossed_rf=crossed_rf,
        observed_ols=observed_ols,
        observed_rf=observed_rf,
        observed_rf_folding_seeds=observed_rf_folding,
        observed_rf_modeling_seeds=observed_rf_modeling,
        diagnostics=diagnostics,
        metadata=metadata,
    )
    write_fixed_seed_references(output_dir.parent, features, target, settings)
    logger.info(
        "All outputs validated and written in %.1f seconds",
        time.monotonic() - started,
    )


if __name__ == "__main__":
    main()
