"""Run crossed seed/data diagnostics for the MNIST convolutional network.

The S5 experiment independently bootstraps the fixed MNIST train and test
partitions, treats the pair as one external dataset resample, and reuses each
pair across every model seed. A separate observed-data seed sweep supplies
the 1,000 values used by the figures and is never used in the S5 components.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import shutil
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from joblib import Parallel, delayed, parallel_config

try:  # Package import and direct execution are both supported.
    from .reporting_utils import (
        S5Diagnostics,
        configure_run_logger,
        crossed_s5_diagnostics,
        format_observed_seed_report,
        format_s5_report,
        load_seed_list,
        observed_seed_summary,
        partitioned_bootstrap_index_blocks,
        resolve_batch_size,
        resolve_worker_count,
        seed_component_blocks,
        validate_s5_log,
    )
except ImportError:  # pragma: no cover - direct CLI execution.
    from reporting_utils import (  # type: ignore[no-redef]
        S5Diagnostics,
        configure_run_logger,
        crossed_s5_diagnostics,
        format_observed_seed_report,
        format_s5_report,
        load_seed_list,
        observed_seed_summary,
        partitioned_bootstrap_index_blocks,
        resolve_batch_size,
        resolve_worker_count,
        seed_component_blocks,
        validate_s5_log,
    )


IMAGE_MAGIC = 2051
LABEL_MAGIC = 2049
MNIST_MEAN = 0.1307
MNIST_SD = 0.3081
ESTIMATED_CPU_WORKER_BYTES = 2 * 1024**3
ESTIMAND_TEMPLATE = "correct predictions out of {n_test} (legacy column: correct)"


@dataclass(frozen=True)
class Settings:
    n_bootstrap_resamples: int = 100
    n_seed_vectors: int = 100
    n_visualization_runs: int = 1_000
    batch_size: int = 1_000
    test_batch_size: int = 1_000
    epochs: int = 1
    learning_rate: float = 1.0
    gamma: float = 0.7  # CLI compatibility; the legacy scheduler was disabled.
    n_jobs: int = -1
    torch_threads_per_worker: int = 4
    checkpoint_cell_batch_size: int = 0
    device: str = "auto"
    deterministic: bool = True
    shuffle_training: bool = True
    max_train_batches: int | None = None

    def validate(self) -> None:
        if self.n_bootstrap_resamples < 2 or self.n_seed_vectors < 2:
            raise ValueError("S5 diagnostics require at least two rows and columns")
        if self.n_visualization_runs < 2:
            raise ValueError("n_visualization_runs must be at least two")
        if min(self.batch_size, self.test_batch_size, self.epochs) < 1:
            raise ValueError("batch sizes and epochs must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(self.gamma) or self.gamma <= 0:
            raise ValueError("gamma must be positive and finite")
        resolve_worker_count(self.n_jobs)
        if self.torch_threads_per_worker < 1:
            raise ValueError("torch_threads_per_worker must be positive")
        if self.checkpoint_cell_batch_size < 0:
            raise ValueError("checkpoint_cell_batch_size cannot be negative")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        if self.max_train_batches is not None and self.max_train_batches < 1:
            raise ValueError("max_train_batches must be positive when supplied")


@dataclass(frozen=True)
class SeedPlan:
    modeling: np.ndarray
    bootstrap: np.ndarray


@dataclass(frozen=True)
class MnistData:
    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray


class Net(nn.Module):
    """The architecture used by the original PyTorch MNIST example."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9_216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = F.relu(self.conv1(values))
        values = F.relu(self.conv2(values))
        values = F.max_pool2d(values, 2)
        values = self.dropout1(values)
        values = torch.flatten(values, 1)
        values = F.relu(self.fc1(values))
        values = self.dropout2(values)
        return F.log_softmax(self.fc2(values), dim=1)


def make_seed_plan(seed_list: Sequence[int], settings: Settings) -> SeedPlan:
    """Allocate disjoint model and external-bootstrap seed blocks."""

    n_vectors = max(settings.n_seed_vectors, settings.n_visualization_runs)
    modeling = seed_component_blocks(
        seed_list, n_vectors=n_vectors, component_names=("modeling",)
    )["modeling"]
    bootstrap_stop = n_vectors + settings.n_bootstrap_resamples
    if bootstrap_stop > len(seed_list):
        raise ValueError(
            f"seed list has {len(seed_list)} entries but {bootstrap_stop} are needed"
        )
    bootstrap = np.asarray(
        seed_list[n_vectors:bootstrap_stop], dtype=np.uint64
    )
    return SeedPlan(modeling=modeling, bootstrap=bootstrap)


def _open_idx(path: Path) -> BinaryIO:
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def _resolve_idx_path(data_dir: Path, filename: str) -> Path:
    uncompressed = data_dir / filename
    compressed = data_dir / f"{filename}.gz"
    if uncompressed.is_file():
        return uncompressed
    if compressed.is_file():
        return compressed
    raise FileNotFoundError(
        f"MNIST IDX file is missing: expected {uncompressed} or {compressed}"
    )


def read_idx_images(path: str | Path) -> np.ndarray:
    """Read and normalize an IDX image file without requiring torchvision."""

    image_path = Path(path)
    with _open_idx(image_path) as handle:
        header = handle.read(16)
        if len(header) != 16:
            raise ValueError(f"truncated IDX image header: {image_path}")
        magic, count, rows, columns = struct.unpack(">IIII", header)
        if magic != IMAGE_MAGIC or (rows, columns) != (28, 28):
            raise ValueError(f"invalid MNIST image header: {image_path}")
        payload = handle.read()
    expected = count * rows * columns
    if len(payload) != expected:
        raise ValueError(
            f"IDX image payload has {len(payload)} bytes; expected {expected}"
        )
    images = np.frombuffer(payload, dtype=np.uint8).reshape(count, 1, rows, columns)
    normalized = images.astype(np.float32)
    normalized /= 255.0
    normalized -= MNIST_MEAN
    normalized /= MNIST_SD
    return normalized


def read_idx_labels(path: str | Path) -> np.ndarray:
    label_path = Path(path)
    with _open_idx(label_path) as handle:
        header = handle.read(8)
        if len(header) != 8:
            raise ValueError(f"truncated IDX label header: {label_path}")
        magic, count = struct.unpack(">II", header)
        if magic != LABEL_MAGIC:
            raise ValueError(f"invalid MNIST label header: {label_path}")
        payload = handle.read()
    if len(payload) != count:
        raise ValueError(f"IDX label payload has {len(payload)} bytes; expected {count}")
    labels = np.frombuffer(payload, dtype=np.uint8).astype(np.int64)
    if np.any(labels > 9):
        raise ValueError(f"MNIST labels must lie in [0, 9]: {label_path}")
    return labels


def load_mnist_data(data_dir: str | Path) -> tuple[MnistData, dict[str, Path]]:
    root = Path(data_dir)
    paths = {
        "train_images": _resolve_idx_path(root, "train-images-idx3-ubyte"),
        "train_labels": _resolve_idx_path(root, "train-labels-idx1-ubyte"),
        "test_images": _resolve_idx_path(root, "t10k-images-idx3-ubyte"),
        "test_labels": _resolve_idx_path(root, "t10k-labels-idx1-ubyte"),
    }
    data = MnistData(
        train_images=read_idx_images(paths["train_images"]),
        train_labels=read_idx_labels(paths["train_labels"]),
        test_images=read_idx_images(paths["test_images"]),
        test_labels=read_idx_labels(paths["test_labels"]),
    )
    if len(data.train_images) != len(data.train_labels):
        raise ValueError("MNIST training images and labels have different lengths")
    if len(data.test_images) != len(data.test_labels):
        raise ValueError("MNIST test images and labels have different lengths")
    return data, paths


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def _available_memory_bytes() -> int | None:
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def resolve_mnist_workers(settings: Settings, device: torch.device) -> int:
    """Resolve safe outer parallelism without CPU/thread or RAM oversubscription."""

    if device.type != "cpu":
        return 1
    requested = resolve_worker_count(settings.n_jobs)
    cpu_workers = max(
        1, resolve_worker_count(-1) // settings.torch_threads_per_worker
    )
    available = _available_memory_bytes()
    memory_workers = (
        requested
        if available is None
        else max(1, int(available * 0.70 // ESTIMATED_CPU_WORKER_BYTES))
    )
    return max(1, min(requested, cpu_workers, memory_workers))


def _checkpoint_wave_size(
    n_items: int, settings: Settings, workers: int, device: torch.device
) -> int:
    if settings.checkpoint_cell_batch_size > 0:
        return min(n_items, settings.checkpoint_cell_batch_size)
    if device.type != "cpu":
        return min(n_items, 16)
    return resolve_batch_size(n_items, batch_size=0, n_jobs=workers)


def _configure_torch_process(settings: Settings) -> None:
    torch.set_num_threads(settings.torch_threads_per_worker)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # PyTorch permits setting this only before inter-op work starts.
    torch.use_deterministic_algorithms(settings.deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = settings.deterministic


def _tensor_batch(
    values: np.ndarray, indices: np.ndarray, device: torch.device
) -> torch.Tensor:
    selected = np.ascontiguousarray(values[indices])
    return torch.from_numpy(selected).to(device=device, non_blocking=False)


def train_and_evaluate(
    data: MnistData,
    *,
    modeling_seed: int,
    train_indices: np.ndarray | None,
    test_indices: np.ndarray | None,
    settings: Settings,
    device_name: str,
) -> int:
    """Train one seeded CNN and return correct predictions on its test sample."""

    _configure_torch_process(settings)
    device = torch.device(device_name)
    seed = int(modeling_seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    train_index = (
        np.arange(len(data.train_labels), dtype=np.int64)
        if train_indices is None
        else np.asarray(train_indices, dtype=np.int64)
    )
    test_index = (
        np.arange(len(data.test_labels), dtype=np.int64)
        if test_indices is None
        else np.asarray(test_indices, dtype=np.int64)
    )
    if len(train_index) == 0 or len(test_index) == 0:
        raise ValueError("training and test index blocks must be non-empty")

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=settings.learning_rate)
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(seed)
    model.train()
    for _epoch in range(settings.epochs):
        if settings.shuffle_training:
            order = torch.randperm(
                len(train_index), generator=shuffle_generator
            ).numpy()
            ordered_indices = train_index[order]
        else:
            ordered_indices = train_index
        for batch_number, start in enumerate(
            range(0, len(ordered_indices), settings.batch_size)
        ):
            if (
                settings.max_train_batches is not None
                and batch_number >= settings.max_train_batches
            ):
                break
            batch_indices = ordered_indices[start : start + settings.batch_size]
            images = _tensor_batch(data.train_images, batch_indices, device)
            labels = _tensor_batch(data.train_labels, batch_indices, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    with torch.inference_mode():
        for start in range(0, len(test_index), settings.test_batch_size):
            batch_indices = test_index[start : start + settings.test_batch_size]
            images = _tensor_batch(data.test_images, batch_indices, device)
            labels = _tensor_batch(data.test_labels, batch_indices, device)
            correct += int((model(images).argmax(dim=1) == labels).sum().item())
    return correct


def _evaluate_crossed_cell(
    flat_index: int,
    data: MnistData,
    train_bootstrap_indices: np.ndarray,
    test_bootstrap_indices: np.ndarray,
    modeling_seeds: np.ndarray,
    n_seed_vectors: int,
    settings: Settings,
    device_name: str,
) -> tuple[int, float]:
    bootstrap_number, seed_number = divmod(flat_index, n_seed_vectors)
    value = train_and_evaluate(
        data,
        modeling_seed=int(modeling_seeds[seed_number]),
        train_indices=train_bootstrap_indices[bootstrap_number],
        test_indices=test_bootstrap_indices[bootstrap_number],
        settings=settings,
        device_name=device_name,
    )
    return flat_index, float(value)


def _evaluate_observed_seed(
    seed_number: int,
    data: MnistData,
    modeling_seeds: np.ndarray,
    settings: Settings,
    device_name: str,
) -> tuple[int, float]:
    value = train_and_evaluate(
        data,
        modeling_seed=int(modeling_seeds[seed_number]),
        train_indices=None,
        test_indices=None,
        settings=settings,
        device_name=device_name,
    )
    return seed_number, float(value)


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npy")
    np.save(temporary, values)
    temporary.replace(path)


def _load_partial_array(
    path: Path, shape: tuple[int, ...], *, maximum_score: int
) -> np.ndarray:
    values = np.asarray(np.load(path), dtype=float)
    if values.shape != shape:
        raise ValueError(f"checkpoint {path} has shape {values.shape}; expected {shape}")
    if np.isinf(values).any():
        raise ValueError(f"checkpoint {path} contains infinite values")
    finite = values[np.isfinite(values)]
    if (
        np.any(finite < 0)
        or np.any(finite > maximum_score)
        or np.any(finite != np.floor(finite))
    ):
        raise ValueError(
            f"checkpoint {path} contains invalid correct-prediction counts; "
            f"finite values must be integers in [0, {maximum_score}]"
        )
    return values.copy()


def run_crossed_grid(
    data: MnistData,
    bootstrap_indices: dict[str, np.ndarray],
    seed_plan: SeedPlan,
    settings: Settings,
    *,
    output_dir: Path,
    logger: Any,
    device: torch.device,
    resume: bool,
) -> np.ndarray:
    """Evaluate independent cells in worker waves and checkpoint each wave."""

    shape = (settings.n_bootstrap_resamples, settings.n_seed_vectors)
    checkpoint_path = output_dir / "mnist_crossed_correct_scores_checkpoint.npy"
    matrix = (
        _load_partial_array(
            checkpoint_path, shape, maximum_score=len(data.test_labels)
        )
        if resume and checkpoint_path.is_file()
        else np.full(shape, np.nan, dtype=float)
    )
    pending = np.flatnonzero(~np.isfinite(matrix.ravel()))
    if pending.size == 0:
        logger.info("Crossed checkpoint already contains all %d cells", matrix.size)
        return matrix

    workers = resolve_mnist_workers(settings, device)
    wave_size = _checkpoint_wave_size(len(pending), settings, workers, device)
    started = time.monotonic()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        with Parallel(n_jobs=workers, max_nbytes="128K", mmap_mode="r") as parallel:
            for start in range(0, len(pending), wave_size):
                current = pending[start : start + wave_size]
                results = parallel(
                    delayed(_evaluate_crossed_cell)(
                        int(flat_index),
                        data,
                        bootstrap_indices["train"],
                        bootstrap_indices["test"],
                        seed_plan.modeling,
                        settings.n_seed_vectors,
                        settings,
                        str(device),
                    )
                    for flat_index in current
                )
                flat = matrix.ravel()
                for flat_index, value in results:
                    flat[flat_index] = value
                _atomic_save_array(checkpoint_path, matrix)
                logger.info(
                    "Crossed grid: %d/%d cells complete (%.1f seconds elapsed)",
                    int(np.isfinite(matrix).sum()),
                    matrix.size,
                    time.monotonic() - started,
                )
    if not np.isfinite(matrix).all():
        raise RuntimeError("crossed MNIST grid finished with incomplete cells")
    return matrix


def run_observed_seed_sweep(
    data: MnistData,
    seed_plan: SeedPlan,
    settings: Settings,
    *,
    output_dir: Path,
    logger: Any,
    device: torch.device,
    resume: bool,
) -> np.ndarray:
    shape = (settings.n_visualization_runs,)
    checkpoint_path = output_dir / "mnist_observed_correct_scores_checkpoint.npy"
    values = (
        _load_partial_array(
            checkpoint_path, shape, maximum_score=len(data.test_labels)
        )
        if resume and checkpoint_path.is_file()
        else np.full(shape, np.nan, dtype=float)
    )
    pending = np.flatnonzero(~np.isfinite(values))
    if pending.size == 0:
        logger.info("Observed checkpoint already contains all %d runs", len(values))
        return values
    workers = resolve_mnist_workers(settings, device)
    wave_size = _checkpoint_wave_size(len(pending), settings, workers, device)
    started = time.monotonic()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        with Parallel(n_jobs=workers, max_nbytes="128K", mmap_mode="r") as parallel:
            for start in range(0, len(pending), wave_size):
                current = pending[start : start + wave_size]
                results = parallel(
                    delayed(_evaluate_observed_seed)(
                        int(seed_number),
                        data,
                        seed_plan.modeling,
                        settings,
                        str(device),
                    )
                    for seed_number in current
                )
                for seed_number, value in results:
                    values[seed_number] = value
                _atomic_save_array(checkpoint_path, values)
                logger.info(
                    "Observed-data sweep: %d/%d runs complete (%.1f seconds elapsed)",
                    int(np.isfinite(values).sum()),
                    len(values),
                    time.monotonic() - started,
                )
    if not np.isfinite(values).all():
        raise RuntimeError("observed MNIST sweep finished with incomplete runs")
    return values


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
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _score_provenance(
    *,
    settings: Settings,
    source_paths: dict[str, Path],
    seed_list_path: Path,
    seed_plan: SeedPlan,
    device: torch.device,
) -> dict[str, Any]:
    score_settings = {
        "n_bootstrap_resamples": settings.n_bootstrap_resamples,
        "n_seed_vectors": settings.n_seed_vectors,
        "n_visualization_runs": settings.n_visualization_runs,
        "batch_size": settings.batch_size,
        "test_batch_size": settings.test_batch_size,
        "epochs": settings.epochs,
        "learning_rate": settings.learning_rate,
        "torch_threads_per_worker": settings.torch_threads_per_worker,
        "resolved_device": str(device),
        "deterministic": settings.deterministic,
        "shuffle_training": settings.shuffle_training,
        "max_train_batches": settings.max_train_batches,
    }
    return {
        # Execution-only controls (n_jobs and checkpoint wave size) are
        # intentionally excluded so a safe resume can tune scheduling.
        "score_settings": score_settings,
        "source_sha256": {
            name: _sha256(path) for name, path in source_paths.items()
        },
        "seed_list_sha256": _sha256(seed_list_path),
        "modeling_seeds": seed_plan.modeling.tolist(),
        "bootstrap_seeds": seed_plan.bootstrap.tolist(),
        "score_implementation_sha256": _sha256(Path(__file__).resolve()),
        "PyTorch": torch.__version__,
        "NumPy": np.__version__,
    }


def _prepare_provenance(
    path: Path, provenance: dict[str, Any], *, require_match: bool
) -> None:
    safe = _json_safe(provenance)
    if require_match:
        if not path.is_file():
            raise FileNotFoundError(
                f"cannot resume/reuse without score provenance: {path}"
            )
        with path.open(encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved != safe:
            raise ValueError(
                "saved MNIST score provenance does not match this run; start "
                "fresh or use a different output directory"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(safe, handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_outputs(
    *,
    output_dir: Path,
    bootstrap_indices: dict[str, np.ndarray],
    seed_plan: SeedPlan,
    crossed_scores: np.ndarray,
    observed_scores: np.ndarray,
    manual_scores: dict[int, int],
    diagnostics: S5Diagnostics,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_save_array(output_dir / "mnist_crossed_correct_scores.npy", crossed_scores)
    _atomic_save_array(output_dir / "mnist_observed_correct_scores.npy", observed_scores)
    _atomic_save_array(
        output_dir / "mnist_bootstrap_train_indices.npy", bootstrap_indices["train"]
    )
    _atomic_save_array(
        output_dir / "mnist_bootstrap_test_indices.npy", bootstrap_indices["test"]
    )

    B, m = crossed_scores.shape
    bootstrap_number = np.repeat(np.arange(B), m)
    seed_number = np.tile(np.arange(m), B)
    n_test = bootstrap_indices["test"].shape[1]
    crossed_frame = pd.DataFrame(
        {
            "Bootstrap_Replicate": bootstrap_number,
            "Seed_Vector": seed_number,
            "Bootstrap_Seed": seed_plan.bootstrap[bootstrap_number],
            "Modeling_Seed": seed_plan.modeling[seed_number],
            "correct": crossed_scores.ravel().astype(np.int64),
            "accuracy": crossed_scores.ravel() / n_test,
        }
    )
    crossed_frame.to_csv(output_dir / "mnist_crossed_bootstrap_scores.csv", index=False)

    observed_frame = pd.DataFrame(
        {
            "Seed_Vector": np.arange(len(observed_scores)),
            "Modeling_Seed": seed_plan.modeling[: len(observed_scores)],
            "correct": observed_scores.astype(np.int64),
            "accuracy": observed_scores / n_test,
        }
    )
    visualization_path = output_dir / "mnist_results.csv"
    observed_frame.to_csv(visualization_path, index=False)
    shutil.copyfile(visualization_path, output_dir / "mnist_visualization_runs.csv")
    pd.DataFrame(
        {
            "Modeling_Seed": [42, 123],
            "correct": [manual_scores[42], manual_scores[123]],
            "accuracy": [manual_scores[42] / n_test, manual_scores[123] / n_test],
        }
    ).to_csv(output_dir / "mnist_results_manual_seeds.csv", index=False)
    pd.DataFrame(
        {
            "Bootstrap_Replicate": np.arange(B),
            "Bootstrap_Seed": seed_plan.bootstrap[:B],
        }
    ).to_csv(output_dir / "mnist_bootstrap_plan.csv", index=False)
    pd.DataFrame(
        {
            "Seed_Vector": np.arange(len(seed_plan.modeling)),
            "Modeling_Seed": seed_plan.modeling,
        }
    ).to_csv(output_dir / "mnist_seed_plan.csv", index=False)
    pd.DataFrame(
        [{"Estimand": metadata["estimand"], **diagnostics.to_dict()}]
    ).to_csv(output_dir / "mnist_s5_diagnostics.csv", index=False)
    payload = {
        "metadata": metadata,
        "diagnostics": {metadata["estimand"]: diagnostics.to_dict()},
        "observed_data_seed_summary": observed_seed_summary(observed_scores).to_dict(),
    }
    with (output_dir / "mnist_s5_diagnostics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(_json_safe(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")


def _build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the crossed MNIST S5 experiment and a separate no-bootstrap "
            "visualisation sweep. IDX files are read directly; torchvision is "
            "not required."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data/MNIST/raw")
    parser.add_argument("--seed-list", type=Path, default=repo_root / "assets/seed_list.txt")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "data/MNIST/results")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=repo_root / "results/results_logs/mnist_s5_diagnostics.log",
    )
    parser.add_argument("--n-bootstrap-resamples", type=int, default=100)
    parser.add_argument("--n-seed-vectors", type=int, default=100)
    parser.add_argument("--n-visualization-runs", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--test-batch-size", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="outer model workers; -1 requests automatic CPU parallelism",
    )
    parser.add_argument("--torch-threads-per-worker", type=int, default=4)
    parser.add_argument(
        "--checkpoint-cell-batch-size", type=int, default=0,
        help="completed cells per checkpoint; 0 uses one resolved worker wave",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--allow-nondeterministic", action="store_true")
    parser.add_argument(
        "--max-train-batches", type=int,
        help="diagnostic/smoke-run limit; omit for the requested full epoch",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume finite cells from crossed and observed checkpoint arrays",
    )
    parser.add_argument(
        "--reuse-crossed-scores", action="store_true",
        help="load a complete mnist_crossed_correct_scores.npy from output-dir",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="run a 2x2 grid and four observed seeds with one training batch",
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
        args.max_train_batches = 1
    settings = Settings(
        n_bootstrap_resamples=args.n_bootstrap_resamples,
        n_seed_vectors=args.n_seed_vectors,
        n_visualization_runs=args.n_visualization_runs,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        gamma=args.gamma,
        n_jobs=args.n_jobs,
        torch_threads_per_worker=args.torch_threads_per_worker,
        checkpoint_cell_batch_size=args.checkpoint_cell_batch_size,
        device=args.device,
        deterministic=not args.allow_nondeterministic,
        shuffle_training=not args.no_shuffle,
        max_train_batches=args.max_train_batches,
    )
    settings.validate()
    output_dir = args.output_dir.resolve()
    log_path = args.log_path.resolve()
    data, source_paths = load_mnist_data(args.data_dir)
    seed_list = load_seed_list(args.seed_list)
    seed_plan = make_seed_plan(seed_list, settings)
    bootstrap_indices = partitioned_bootstrap_index_blocks(
        {"train": len(data.train_labels), "test": len(data.test_labels)},
        seed_plan.bootstrap[: settings.n_bootstrap_resamples],
    )
    device = _resolve_device(settings.device)
    workers = resolve_mnist_workers(settings, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = _score_provenance(
        settings=settings,
        source_paths=source_paths,
        seed_list_path=args.seed_list.resolve(),
        seed_plan=seed_plan,
        device=device,
    )
    _prepare_provenance(
        output_dir / "mnist_score_provenance.json",
        provenance,
        require_match=bool(args.resume or args.reuse_crossed_scores),
    )
    # Do not truncate an existing valid report until a resume/reuse request
    # has passed its score-provenance check.
    logger = configure_run_logger("mnist_s5", log_path)
    logger.info("Starting MNIST S5 analysis")
    logger.info("MNIST data directory: %s", args.data_dir.resolve())
    logger.info("Seed list: %s", args.seed_list.resolve())
    logger.info("Run-level output directory: %s", output_dir)
    _atomic_save_array(
        output_dir / "mnist_bootstrap_train_indices.npy", bootstrap_indices["train"]
    )
    _atomic_save_array(
        output_dir / "mnist_bootstrap_test_indices.npy", bootstrap_indices["test"]
    )
    logger.info(
        "Prepared %d train and %d test images; %d shared partition-preserving "
        "bootstrap pairs crossed with %d model seeds (%d cells)",
        len(data.train_labels), len(data.test_labels),
        settings.n_bootstrap_resamples, settings.n_seed_vectors,
        settings.n_bootstrap_resamples * settings.n_seed_vectors,
    )
    logger.info(
        "Device: %s; outer workers: %d; Torch threads per worker: %d; "
        "checkpoint cell batch: %d",
        device, workers, settings.torch_threads_per_worker,
        _checkpoint_wave_size(
            settings.n_bootstrap_resamples * settings.n_seed_vectors,
            settings,
            workers,
            device,
        ),
    )

    final_crossed_path = output_dir / "mnist_crossed_correct_scores.npy"
    expected_shape = (settings.n_bootstrap_resamples, settings.n_seed_vectors)
    if args.reuse_crossed_scores:
        crossed_scores = _load_partial_array(
            final_crossed_path,
            expected_shape,
            maximum_score=len(data.test_labels),
        )
        if not np.isfinite(crossed_scores).all():
            raise ValueError("reused crossed score matrix must be complete")
        logger.info("Reused %d completed crossed cells", crossed_scores.size)
    else:
        crossed_scores = run_crossed_grid(
            data, bootstrap_indices, seed_plan, settings,
            output_dir=output_dir, logger=logger, device=device, resume=args.resume,
        )
    observed_scores = run_observed_seed_sweep(
        data, seed_plan, settings,
        output_dir=output_dir, logger=logger, device=device, resume=args.resume,
    )
    manual_scores = {
        seed: train_and_evaluate(
            data, modeling_seed=seed, train_indices=None, test_indices=None,
            settings=settings, device_name=str(device),
        )
        for seed in (42, 123)
    }

    diagnostics = crossed_s5_diagnostics(crossed_scores)
    estimand = ESTIMAND_TEMPLATE.format(n_test=len(data.test_labels))
    details = {
        "within_cell_replications": "1 complete CNN training/evaluation",
        "external_dataset_bootstrap": (
            "train and test partitions resampled independently with replacement; "
            "partition roles and sizes retained"
        ),
        "bootstrap_design": (
            "crossed; each train/test resample pair is reused for every model seed"
        ),
        "bootstrap_PRNG": (
            f"NumPy SeedSequence child streams with PCG64 {np.__version__}"
        ),
        "algorithmic_PRNG": (
            f"torch.manual_seed; PyTorch {torch.__version__}; one scalar seed "
            "controls initialization, dropout, and an explicit CPU randperm stream"
        ),
        "seed_components": "joint model-initialization/dropout/minibatch-order seed",
        "seed_components_varied": "jointly",
        "model": (
            "two-convolution PyTorch CNN with Adadelta"
            f"(lr={settings.learning_rate}), epochs={settings.epochs}"
        ),
        "training_batch_size": settings.batch_size,
        "test_batch_size": settings.test_batch_size,
        "training_shuffle": settings.shuffle_training,
        "test_shuffle": False,
        "shuffle_implementation_note": (
            "explicit on every device to match manuscript S3.5; the legacy "
            "script enabled shuffle only on CUDA"
        ),
        "learning_rate_scheduler": (
            "disabled, matching the commented-out legacy StepLR call; gamma is "
            "accepted only for CLI compatibility"
        ),
        "normalization": f"mean={MNIST_MEAN}, sd={MNIST_SD}",
        "deterministic_algorithms": settings.deterministic,
        "device": str(device),
        "parallel_backend": (
            "joblib loky cell workers on CPU; accelerator execution is serial"
        ),
        "parallel_workers": workers,
        "torch_threads_per_worker": settings.torch_threads_per_worker,
        "checkpoint_unit": "independent bootstrap-by-seed cells",
        "max_train_batches": (
            "full epoch" if settings.max_train_batches is None
            else settings.max_train_batches
        ),
        "Python_version": platform.python_version(),
        "NumPy_version": np.__version__,
        "joblib_version": joblib.__version__,
    }
    logger.info(
        "\n%s",
        format_s5_report(
            diagnostics, estimand=estimand, computational_details=details
        ),
    )
    logger.info(
        "\n%s",
        format_observed_seed_report(
            observed_seed_summary(observed_scores), estimand=estimand
        ),
    )
    logger.info(
        "Seed-averaged crossed accuracy fraction=%.12g; observed-data accuracy "
        "fraction=%.12g",
        diagnostics.seed_averaged_estimate / len(data.test_labels),
        float(observed_scores.mean()) / len(data.test_labels),
    )

    metadata = {
        "analysis": "MNIST convolutional neural-network seed sensitivity",
        "estimand": estimand,
        "settings": asdict(settings),
        "resolved_device": str(device),
        "resolved_parallel_workers": workers,
        "n_train": len(data.train_labels),
        "n_test": len(data.test_labels),
        "source_files": {name: str(path) for name, path in source_paths.items()},
        "source_sha256": provenance["source_sha256"],
        "seed_list": str(args.seed_list.resolve()),
        "seed_list_sha256": provenance["seed_list_sha256"],
        "crossed_scores_reused": bool(args.reuse_crossed_scores),
        "resumed_from_checkpoints": bool(args.resume),
        "computational_details": details,
        "versions": {
            "Python": platform.python_version(), "NumPy": np.__version__,
            "pandas": pd.__version__, "PyTorch": torch.__version__,
            "joblib": joblib.__version__,
        },
    }
    write_outputs(
        output_dir=output_dir, bootstrap_indices=bootstrap_indices,
        seed_plan=seed_plan, crossed_scores=crossed_scores,
        observed_scores=observed_scores, manual_scores=manual_scores,
        diagnostics=diagnostics, metadata=metadata,
    )
    validate_s5_log(log_path, (estimand,))
    logger.info(
        "MNIST analysis complete in %.1f seconds; all outputs and the six-item "
        "S5 log contract validated",
        time.monotonic() - started,
    )


if __name__ == "__main__":
    # Loky must be able to import the PyTorch model/worker definitions by
    # module name.  Calling an importable copy avoids cloudpickle trying to
    # serialize PyTorch's internal (unpicklable) cuDNN module from __main__.
    import importlib

    executable_module = (
        f"{__package__}.build_mnist_seeds" if __package__ else "build_mnist_seeds"
    )
    importlib.import_module(executable_module).main()
