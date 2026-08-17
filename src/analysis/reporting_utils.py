"""Reusable reporting helpers for seed-sensitivity experiments.

The central routine implements the crossed bootstrap-by-seed estimators in
Supplementary Sections S4.2.1 and S5 of the manuscript.  Rows of the input
matrix are shared bootstrap resamples and columns are algorithmic seed
vectors.  This orientation is deliberately explicit because a nested design
does not identify the same variance components.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from joblib import effective_n_jobs


MAX_UINT32 = 2**32 - 1


@dataclass(frozen=True)
class S5Diagnostics:
    """S5 estimates from a complete crossed ``B x m`` score matrix.

    The raw equation-(27) method-of-moments estimate is retained in
    ``between_seed_variance_raw``.  Because a variance cannot be negative,
    the S5 point estimates use the explicitly reported nonnegative boundary
    value ``max(raw, 0)`` in ``between_seed_variance``.
    """

    n_bootstrap_resamples: int
    n_seed_vectors: int
    seed_averaged_estimate: float
    data_variance: float
    data_uncertainty_sd: float
    between_seed_variance_raw: float
    between_seed_variance: float
    variance_component_boundary_hit: bool
    between_seed_variability_sd: float | None
    relative_importance: float | None
    algorithmic_variance_share: float | None
    total_order_algorithmic_variance_share: float | None
    data_main_effect_variance: float
    data_seed_interaction_variance: float
    seed_mean_square: float
    data_mean_square: float
    interaction_mean_square: float

    def to_dict(self) -> dict[str, int | float | None]:
        """Return a serialization-friendly flat mapping."""

        return asdict(self)


@dataclass(frozen=True)
class ObservedSeedSummary:
    """Allowed summary when seeds vary on one observed dataset only."""

    n_seed_vectors: int
    observed_data_seed_average: float
    observed_between_seed_sd: float
    minimum: float
    maximum: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def load_seed_list(path: str | Path) -> list[int]:
    """Read a newline-delimited seed list and validate uint32 compatibility."""

    seed_path = Path(path)
    seeds: list[int] = []
    with seed_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            try:
                seed = int(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid seed on line {line_number} of {seed_path}"
                ) from exc
            if not 0 <= seed <= MAX_UINT32:
                raise ValueError(
                    f"seed on line {line_number} of {seed_path} is outside "
                    f"[0, {MAX_UINT32}]"
                )
            seeds.append(seed)

    if not seeds:
        raise ValueError(f"seed list is empty: {seed_path}")
    return seeds


def seed_component_blocks(
    seeds: Sequence[int],
    *,
    n_vectors: int,
    component_names: Sequence[str],
    offset: int = 0,
) -> dict[str, np.ndarray]:
    """Allocate disjoint, stage-major seed blocks to seed-vector components.

    For example, two components and 1,000 vectors consume the first 1,000
    selected values for component one and the next 1,000 for component two.
    The returned arrays align by position to form 1,000 jointly varied seed
    vectors.
    """

    names = tuple(component_names)
    if n_vectors < 1:
        raise ValueError("n_vectors must be positive")
    if offset < 0:
        raise ValueError("offset cannot be negative")
    if not names or len(set(names)) != len(names):
        raise ValueError("component_names must be non-empty and unique")

    required = n_vectors * len(names)
    stop = offset + required
    if stop > len(seeds):
        raise ValueError(
            f"seed list has {len(seeds)} values but positions "
            f"{offset}:{stop} are required"
        )

    selected = np.asarray(seeds[offset:stop], dtype=np.uint64)
    if np.any(selected > MAX_UINT32):
        raise ValueError("all selected seeds must be uint32-compatible")
    return {
        name: selected[index * n_vectors : (index + 1) * n_vectors].copy()
        for index, name in enumerate(names)
    }


def bootstrap_index_block(
    n_observations: int,
    bootstrap_seeds: Sequence[int],
    *,
    sample_size: int | None = None,
) -> np.ndarray:
    """Draw one reproducible bootstrap index row per supplied seed.

    Callers must reuse each row for every algorithmic seed vector.  This
    shared block is what makes the design crossed rather than nested.
    """

    if n_observations < 1:
        raise ValueError("n_observations must be positive")
    if len(bootstrap_seeds) == 0:
        raise ValueError("at least one bootstrap seed is required")
    draw_size = n_observations if sample_size is None else sample_size
    if draw_size < 1:
        raise ValueError("sample_size must be positive")

    indices = np.empty((len(bootstrap_seeds), draw_size), dtype=np.int64)
    for row, seed in enumerate(bootstrap_seeds):
        if not 0 <= int(seed) <= MAX_UINT32:
            raise ValueError("bootstrap seeds must be uint32-compatible")
        rng = np.random.Generator(np.random.PCG64(int(seed)))
        indices[row] = rng.integers(
            0, n_observations, size=draw_size, dtype=np.int64
        )
    return indices


def partitioned_bootstrap_index_blocks(
    partition_sizes: Mapping[str, int],
    bootstrap_seeds: Sequence[int],
) -> dict[str, np.ndarray]:
    """Draw independent bootstrap blocks for fixed dataset partitions.

    One ``SeedSequence`` is created per external-bootstrap replicate and
    spawned into one PCG64 child stream per named partition.  Consequently,
    callers can preserve roles such as train/test while still treating the
    pair of resamples as one shared ``D_b*`` row reused across every model
    seed.
    """

    sizes = tuple(partition_sizes.items())
    if not sizes or len({name for name, _ in sizes}) != len(sizes):
        raise ValueError("partition_sizes must have unique named partitions")
    for name, size in sizes:
        if not name or not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError(f"partition size for {name!r} must be a positive integer")
    if len(bootstrap_seeds) == 0:
        raise ValueError("at least one bootstrap seed is required")

    blocks = {
        name: np.empty((len(bootstrap_seeds), size), dtype=np.int64)
        for name, size in sizes
    }
    for row, seed in enumerate(bootstrap_seeds):
        if not 0 <= int(seed) <= MAX_UINT32:
            raise ValueError("bootstrap seeds must be uint32-compatible")
        children = np.random.SeedSequence(int(seed)).spawn(len(sizes))
        for (name, size), child in zip(sizes, children, strict=True):
            rng = np.random.Generator(np.random.PCG64(child))
            blocks[name][row] = rng.integers(0, size, size=size, dtype=np.int64)
    return blocks


def full_factorial_seed_grid(
    first_component: Sequence[int] | np.ndarray,
    second_component: Sequence[int] | np.ndarray,
    *,
    n_runs: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Build a near-square two-component full factorial of exactly ``n_runs``.

    The smaller factor is chosen as the largest divisor no greater than
    ``sqrt(n_runs)``.  For example, 1,000 runs become 40 values of the first
    component crossed with 25 values of the second component.  This is useful
    for visualisation sweeps that must retain within-component replication.
    """

    if n_runs < 1:
        raise ValueError("n_runs must be positive")
    n_second = math.isqrt(n_runs)
    while n_runs % n_second:
        n_second -= 1
    n_first = n_runs // n_second

    first = np.asarray(first_component)
    second = np.asarray(second_component)
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("seed components must be one-dimensional")
    if first.size < n_first or second.size < n_second:
        raise ValueError(
            "seed components are too short for the requested factorial: "
            f"need {n_first} and {n_second}, got {first.size} and {second.size}"
        )

    return (
        np.repeat(first[:n_first], n_second),
        np.tile(second[:n_second], n_first),
        n_first,
        n_second,
    )


def resolve_worker_count(n_jobs: int, *, n_tasks: int | None = None) -> int:
    """Resolve joblib worker syntax and optionally cap it to useful work.

    ``n_jobs=-1`` uses every CPU available to the process. Other negative
    values retain joblib's usual "all CPUs minus N" interpretation.
    """

    if n_jobs == 0:
        raise ValueError("n_jobs cannot be zero")
    workers = effective_n_jobs(n_jobs)
    if n_tasks is not None:
        if n_tasks < 1:
            raise ValueError("n_tasks must be positive")
        workers = min(workers, n_tasks)
    return max(1, workers)


def resolve_batch_size(
    n_items: int, *, batch_size: int, n_jobs: int
) -> int:
    """Resolve a zero (automatic) batch to one wave of parallel work."""

    if n_items < 1:
        raise ValueError("n_items must be positive")
    if batch_size < 0:
        raise ValueError("batch_size cannot be negative")
    if batch_size == 0:
        return resolve_worker_count(n_jobs, n_tasks=n_items)
    return min(n_items, batch_size)


def parallel_chunk_ranges(
    n_items: int,
    *,
    n_jobs: int,
    tasks_per_worker: int = 4,
    max_chunk_size: int = 50,
) -> list[tuple[int, int]]:
    """Create balanced contiguous chunks with enough tasks for load balancing."""

    if n_items < 1:
        raise ValueError("n_items must be positive")
    if tasks_per_worker < 1 or max_chunk_size < 1:
        raise ValueError("chunk controls must be positive")
    workers = resolve_worker_count(n_jobs, n_tasks=n_items)
    target_tasks = min(n_items, workers * tasks_per_worker)
    chunk_size = min(max_chunk_size, math.ceil(n_items / target_tasks))
    return [
        (start, min(start + chunk_size, n_items))
        for start in range(0, n_items, chunk_size)
    ]


def crossed_s5_diagnostics(score_matrix: np.ndarray) -> S5Diagnostics:
    """Compute the six S5 reporting items from a crossed score matrix.

    The score matrix must have shape ``(B, m)`` with bootstrap resamples on
    rows and seed vectors on columns.  Equations (25)--(27) supply the data,
    seed, and interaction variance estimates.  Item 6 (computational details)
    is supplied separately when formatting the report.
    """

    scores = np.asarray(score_matrix, dtype=float)
    if scores.ndim != 2 or min(scores.shape) < 2:
        raise ValueError(
            "score_matrix must be two-dimensional with at least two rows "
            "and two columns"
        )
    if not np.isfinite(scores).all():
        raise ValueError("score_matrix must be complete and finite")

    n_bootstrap, n_seeds = scores.shape
    grand_mean = float(scores.mean())
    bootstrap_means = scores.mean(axis=1)
    seed_means = scores.mean(axis=0)

    # Equation (25): average the sample variance across bootstrap resamples
    # at each fixed seed.  This estimates V_D + V_DS.
    data_variance = float(scores.var(axis=0, ddof=1).mean())

    # Equations (26)--(27): balanced two-way random-effects mean squares.
    seed_ms = float(n_bootstrap * seed_means.var(ddof=1))
    data_ms = float(n_seeds * bootstrap_means.var(ddof=1))
    residual = (
        scores
        - bootstrap_means[:, np.newaxis]
        - seed_means[np.newaxis, :]
        + grand_mean
    )
    interaction_ms = float(
        np.square(residual).sum()
        / ((n_bootstrap - 1) * (n_seeds - 1))
    )

    between_seed_variance_raw = (seed_ms - interaction_ms) / n_bootstrap
    between_seed_variance = max(0.0, between_seed_variance_raw)
    boundary_hit = between_seed_variance_raw < 0
    data_main_effect_variance = (data_ms - interaction_ms) / n_seeds

    # This identity follows from the balanced layout and guards against axis
    # swaps or transcription errors in future reuse.
    if not np.isclose(
        data_variance,
        data_main_effect_variance + interaction_ms,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError("crossed-grid variance identity failed")

    data_sd = math.sqrt(data_variance)
    seed_sd: float | None
    relative: float | None
    first_order_share: float | None
    total_order_share: float | None

    seed_sd = math.sqrt(between_seed_variance)
    if data_sd == 0:
        relative = None if seed_sd == 0 else math.inf
    else:
        relative = seed_sd / data_sd

    denominator = between_seed_variance + data_variance
    if denominator == 0:
        first_order_share = None
        total_order_share = None
    else:
        first_order_share = between_seed_variance / denominator
        # A negative finite-sample V_D estimate makes the raw total-order
        # plug-in exceed one.  Preserve V_D for audit, but do not label an
        # out-of-bounds quantity as a variance share or silently clip it.
        total_order_share = (
            None
            if data_main_effect_variance < 0
            else (between_seed_variance + interaction_ms) / denominator
        )

    return S5Diagnostics(
        n_bootstrap_resamples=n_bootstrap,
        n_seed_vectors=n_seeds,
        seed_averaged_estimate=grand_mean,
        data_variance=data_variance,
        data_uncertainty_sd=data_sd,
        between_seed_variance_raw=between_seed_variance_raw,
        between_seed_variance=between_seed_variance,
        variance_component_boundary_hit=boundary_hit,
        between_seed_variability_sd=seed_sd,
        relative_importance=relative,
        algorithmic_variance_share=first_order_share,
        total_order_algorithmic_variance_share=total_order_share,
        data_main_effect_variance=data_main_effect_variance,
        data_seed_interaction_variance=interaction_ms,
        seed_mean_square=seed_ms,
        data_mean_square=data_ms,
        interaction_mean_square=interaction_ms,
    )


def observed_seed_summary(scores: Sequence[float] | np.ndarray) -> ObservedSeedSummary:
    """Summarise seed variation conditional on one observed dataset."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("scores must be a one-dimensional sequence of length >= 2")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    return ObservedSeedSummary(
        n_seed_vectors=int(values.size),
        observed_data_seed_average=float(values.mean()),
        observed_between_seed_sd=float(values.std(ddof=1)),
        minimum=float(values.min()),
        maximum=float(values.max()),
    )


def _display_number(value: Any) -> str:
    if value is None:
        return "not_estimable"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    return str(value)


def format_s5_report(
    diagnostics: S5Diagnostics,
    *,
    estimand: str,
    computational_details: Mapping[str, Any],
) -> str:
    """Format all six S5 reporting items as an auditable text block."""

    d = diagnostics
    lines = [
        f"S5 recommended reporting and diagnostics: {estimand}",
        "1. Seed-averaged estimate",
        f"   theta_bar={_display_number(d.seed_averaged_estimate)}",
        "2. Data uncertainty",
        f"   SD_theta_D={_display_number(d.data_uncertainty_sd)}",
        f"   sigma_D_squared={_display_number(d.data_variance)}",
        "3. Bias-corrected between-seed variability",
        f"   SD_theta_S={_display_number(d.between_seed_variability_sd)}",
        "   sigma_S_squared_adj_nonnegative="
        f"{_display_number(d.between_seed_variance)}",
        "   sigma_S_squared_adj_raw="
        f"{_display_number(d.between_seed_variance_raw)}",
        "   variance_component_boundary_hit="
        f"{str(d.variance_component_boundary_hit).lower()}",
        "4. Relative importance of algorithmic randomness",
        f"   r={_display_number(d.relative_importance)}",
        "5. Algorithmic variance share",
        f"   rho_S={_display_number(d.algorithmic_variance_share)}",
        "   Total-order algorithmic variance share",
        "   rho_T_S="
        f"{_display_number(d.total_order_algorithmic_variance_share)}",
        "6. Computational details",
        f"   m_seed_vectors={d.n_seed_vectors}",
        f"   B_bootstrap_resamples={d.n_bootstrap_resamples}",
    ]
    lines.extend(
        f"   {key}={_display_number(value)}"
        for key, value in computational_details.items()
    )
    lines.extend(
        [
            "Supporting crossed-grid diagnostics",
            f"   V_D_hat={_display_number(d.data_main_effect_variance)}",
            f"   V_DS_hat={_display_number(d.data_seed_interaction_variance)}",
            f"   MS_S={_display_number(d.seed_mean_square)}",
            f"   MS_D={_display_number(d.data_mean_square)}",
            f"   MS_int={_display_number(d.interaction_mean_square)}",
        ]
    )
    if d.variance_component_boundary_hit:
        lines.append(
            "   warning=The unconstrained equation-(27) seed-variance estimate "
            "is slightly negative. S5 point estimates use the explicitly "
            "reported nonnegative boundary value zero; the raw estimate is "
            "retained above. A boundary point estimate does not prove that the "
            "true seed main-effect variance is exactly zero."
        )
    if d.data_main_effect_variance < 0:
        lines.append(
            "   warning=The finite-sample data main-effect estimate is negative; "
            "the total-order share would lie outside [0, 1], so rho_T_S is not "
            "reported and was not silently clipped."
        )
    return "\n".join(lines)


def format_observed_seed_report(
    summary: ObservedSeedSummary, *, estimand: str
) -> str:
    """Format the restricted S5 report for a no-bootstrap seed sweep."""

    return "\n".join(
        [
            f"Observed-data seed sweep (no bootstrap): {estimand}",
            f"n_seed_vectors={summary.n_seed_vectors}",
            "observed_data_seed_average="
            f"{_display_number(summary.observed_data_seed_average)}",
            "observed_between_seed_sd_conditional_on_D_obs="
            f"{_display_number(summary.observed_between_seed_sd)}",
            f"minimum={_display_number(summary.minimum)}",
            f"maximum={_display_number(summary.maximum)}",
            "interpretation=Conditional variability on the single observed "
            "dataset; it estimates the total seed effect V_S + V_DS.",
            "not_reported=Data uncertainty, r, and rho_S are not identified "
            "without resampling and are not computed from this sweep.",
        ]
    )


def configure_run_logger(name: str, path: str | Path) -> logging.Logger:
    """Create a console-and-file logger, overwriting the named run log."""

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def validate_s5_log(path: str | Path, estimands: Sequence[str]) -> None:
    """Require a complete six-item S5 report block for every estimand."""

    report_path = Path(path)
    text = report_path.read_text(encoding="utf-8")
    headings = (
        "1. Seed-averaged estimate",
        "2. Data uncertainty",
        "3. Bias-corrected between-seed variability",
        "4. Relative importance of algorithmic randomness",
        "5. Algorithmic variance share",
        "6. Computational details",
    )
    markers = [
        f"S5 recommended reporting and diagnostics: {estimand}"
        for estimand in estimands
    ]
    if not markers:
        raise ValueError("at least one estimand is required")
    for marker in markers:
        start = text.find(marker)
        if start < 0:
            raise RuntimeError(f"S5 log is missing estimand block: {marker}")
        following = [
            position
            for other in markers
            if other != marker and (position := text.find(other, start + len(marker))) >= 0
        ]
        stop = min(following, default=len(text))
        block = text[start:stop]
        missing = [heading for heading in headings if heading not in block]
        if missing:
            raise RuntimeError(
                f"S5 log block for {marker!r} is incomplete; missing: "
                + ", ".join(missing)
            )


__all__ = [
    "ObservedSeedSummary",
    "S5Diagnostics",
    "bootstrap_index_block",
    "configure_run_logger",
    "crossed_s5_diagnostics",
    "format_observed_seed_report",
    "format_s5_report",
    "full_factorial_seed_grid",
    "load_seed_list",
    "observed_seed_summary",
    "parallel_chunk_ranges",
    "partitioned_bootstrap_index_blocks",
    "resolve_batch_size",
    "resolve_worker_count",
    "seed_component_blocks",
    "validate_s5_log",
]
