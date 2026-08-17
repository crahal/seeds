"""Focused design, IDX, multiprocessing, and log-contract tests for MNIST."""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "src" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from build_mnist_seeds import (  # noqa: E402
    Settings,
    _load_partial_array,
    load_mnist_data,
    main,
    make_seed_plan,
)


def _write_idx_images(path: Path, values: np.ndarray) -> None:
    header = struct.pack(">IIII", 2051, len(values), 28, 28)
    path.write_bytes(header + np.asarray(values, dtype=np.uint8).tobytes())


def _write_idx_labels(path: Path, values: np.ndarray) -> None:
    header = struct.pack(">II", 2049, len(values))
    path.write_bytes(header + np.asarray(values, dtype=np.uint8).tobytes())


def _write_tiny_mnist(root: Path) -> None:
    rng = np.random.default_rng(2026)
    train = rng.integers(0, 256, size=(12, 1, 28, 28), dtype=np.uint8)
    test = rng.integers(0, 256, size=(6, 1, 28, 28), dtype=np.uint8)
    _write_idx_images(root / "train-images-idx3-ubyte", train)
    _write_idx_labels(root / "train-labels-idx1-ubyte", np.arange(12) % 10)
    _write_idx_images(root / "t10k-images-idx3-ubyte", test)
    _write_idx_labels(root / "t10k-labels-idx1-ubyte", np.arange(6) % 10)


class MnistDesignTests(unittest.TestCase):
    def test_requested_defaults_and_disjoint_seed_stages(self) -> None:
        settings = Settings()
        self.assertEqual(
            (
                settings.n_bootstrap_resamples,
                settings.n_seed_vectors,
                settings.n_visualization_runs,
            ),
            (100, 100, 1_000),
        )
        plan = make_seed_plan(list(range(2_000)), settings)
        self.assertEqual(len(plan.modeling), 1_000)
        self.assertEqual(len(plan.bootstrap), 100)
        self.assertTrue(set(plan.modeling).isdisjoint(set(plan.bootstrap)))

    def test_idx_reader_normalizes_and_preserves_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_tiny_mnist(root)
            data, paths = load_mnist_data(root)
            self.assertEqual(data.train_images.shape, (12, 1, 28, 28))
            self.assertEqual(data.test_images.shape, (6, 1, 28, 28))
            self.assertEqual(data.train_images.dtype, np.float32)
            np.testing.assert_array_equal(data.test_labels, np.arange(6) % 10)
            self.assertEqual(set(paths), {
                "train_images", "train_labels", "test_images", "test_labels"
            })

    def test_partial_score_loader_rejects_fractional_or_out_of_range_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.npy"
            for invalid in (-1.0, 2.5, 7.0):
                with self.subTest(invalid=invalid):
                    np.save(path, np.array([[1.0, invalid], [np.nan, 2.0]]))
                    with self.assertRaises(ValueError):
                        _load_partial_array(path, (2, 2), maximum_score=6)


class MnistCliTests(unittest.TestCase):
    def test_parallel_tiny_run_writes_complete_s5_log_and_plot_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "raw"
            output_dir = root / "results"
            data_dir.mkdir()
            _write_tiny_mnist(data_dir)
            seed_path = root / "seeds.txt"
            seed_path.write_text(
                "".join(f"{seed}\n" for seed in range(100, 120)),
                encoding="utf-8",
            )
            log_path = root / "mnist.log"
            common = [
                "--data-dir", str(data_dir),
                "--seed-list", str(seed_path),
                "--n-bootstrap-resamples", "2",
                "--n-seed-vectors", "2",
                "--n-visualization-runs", "4",
                "--batch-size", "4",
                "--test-batch-size", "6",
                "--max-train-batches", "1",
                "--torch-threads-per-worker", "1",
                "--device", "cpu",
            ]
            main(common + [
                "--output-dir", str(output_dir),
                "--log-path", str(log_path),
                "--n-jobs", "2",
            ])

            crossed = pd.read_csv(
                output_dir / "mnist_crossed_bootstrap_scores.csv"
            )
            observed = pd.read_csv(output_dir / "mnist_results.csv")
            manual = pd.read_csv(output_dir / "mnist_results_manual_seeds.csv")
            self.assertEqual(crossed.shape[0], 4)
            self.assertEqual(observed.shape[0], 4)
            self.assertEqual(manual["Modeling_Seed"].tolist(), [42, 123])
            self.assertIn("correct", observed.columns)
            self.assertEqual(
                np.load(output_dir / "mnist_crossed_correct_scores.npy").shape,
                (2, 2),
            )
            text = log_path.read_text(encoding="utf-8")
            for heading in (
                "1. Seed-averaged estimate",
                "2. Data uncertainty",
                "3. Bias-corrected between-seed variability",
                "4. Relative importance of algorithmic randomness",
                "5. Algorithmic variance share",
                "6. Computational details",
            ):
                self.assertIn(heading, text)
            self.assertIn("S5 log contract validated", text)

            with self.assertRaises(ValueError):
                main(common + [
                    "--output-dir", str(output_dir),
                    "--log-path", str(log_path),
                    "--n-jobs", "1",
                    "--resume",
                    "--batch-size", "5",
                ])
            self.assertEqual(log_path.read_text(encoding="utf-8"), text)

            serial_output = root / "serial-results"
            main(common + [
                "--output-dir", str(serial_output),
                "--log-path", str(root / "serial.log"),
                "--n-jobs", "1",
            ])
            np.testing.assert_array_equal(
                np.load(output_dir / "mnist_crossed_correct_scores.npy"),
                np.load(serial_output / "mnist_crossed_correct_scores.npy"),
            )
            np.testing.assert_array_equal(
                np.load(output_dir / "mnist_observed_correct_scores.npy"),
                np.load(serial_output / "mnist_observed_correct_scores.npy"),
            )


if __name__ == "__main__":
    unittest.main()
