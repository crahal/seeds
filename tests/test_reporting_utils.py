"""Focused tests for the reusable S5 reporting calculations."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "src" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from reporting_utils import (  # noqa: E402
    bootstrap_index_block,
    crossed_s5_diagnostics,
    format_s5_report,
    observed_seed_summary,
    parallel_chunk_ranges,
    partitioned_bootstrap_index_blocks,
    resolve_batch_size,
    resolve_worker_count,
    seed_component_blocks,
    validate_s5_log,
)


class CrossedDiagnosticsTests(unittest.TestCase):
    def test_additive_grid_recovers_known_components(self) -> None:
        data_effect = np.array([-1.0, 0.0, 1.0, 2.0])
        seed_effect = np.array([0.0, 2.0, 4.0])
        scores = data_effect[:, None] + seed_effect[None, :]

        result = crossed_s5_diagnostics(scores)

        self.assertAlmostEqual(result.seed_averaged_estimate, scores.mean())
        self.assertAlmostEqual(
            result.data_variance, data_effect.var(ddof=1)
        )
        self.assertAlmostEqual(
            result.between_seed_variance, seed_effect.var(ddof=1)
        )
        self.assertAlmostEqual(result.data_seed_interaction_variance, 0.0)
        expected_share = seed_effect.var(ddof=1) / (
            seed_effect.var(ddof=1) + data_effect.var(ddof=1)
        )
        self.assertAlmostEqual(result.algorithmic_variance_share, expected_share)
        self.assertAlmostEqual(
            result.total_order_algorithmic_variance_share, expected_share
        )

    def test_components_are_invariant_to_row_and_column_order(self) -> None:
        scores = np.array(
            [[0.2, 0.8, 0.4], [0.7, 0.1, 0.5], [0.3, 0.9, 0.6], [0.4, 0.2, 1.0]]
        )
        expected = crossed_s5_diagnostics(scores)
        actual = crossed_s5_diagnostics(scores[[2, 0, 3, 1]][:, [1, 2, 0]])
        fields = (
            "data_variance",
            "between_seed_variance",
            "data_main_effect_variance",
            "data_seed_interaction_variance",
        )
        for field in fields:
            self.assertAlmostEqual(getattr(actual, field), getattr(expected, field))

    def test_negative_seed_component_uses_explicit_boundary_estimate(self) -> None:
        result = crossed_s5_diagnostics(
            np.array([[1.0, -1.0], [-1.0, 1.0]])
        )
        self.assertLess(result.between_seed_variance_raw, 0)
        self.assertEqual(result.between_seed_variance, 0)
        self.assertTrue(result.variance_component_boundary_hit)
        self.assertEqual(result.between_seed_variability_sd, 0)
        self.assertEqual(result.relative_importance, 0)
        self.assertEqual(result.algorithmic_variance_share, 0)
        report = format_s5_report(
            result, estimand="test", computational_details={}
        )
        self.assertIn("sigma_S_squared_adj_raw=-2", report)
        self.assertIn("variance_component_boundary_hit=true", report)
        self.assertIn("does not prove", report)

    def test_constant_grid_has_undefined_ratio_and_shares(self) -> None:
        result = crossed_s5_diagnostics(np.ones((3, 4)))
        self.assertEqual(result.data_uncertainty_sd, 0)
        self.assertEqual(result.between_seed_variability_sd, 0)
        self.assertIsNone(result.relative_importance)
        self.assertIsNone(result.algorithmic_variance_share)
        self.assertIsNone(result.total_order_algorithmic_variance_share)

    def test_negative_data_component_invalidates_only_total_order_share(self) -> None:
        interaction = np.array(
            [[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        )
        scores = interaction + np.array([-2.0, 0.0, 2.0])[None, :]
        result = crossed_s5_diagnostics(scores)
        self.assertGreaterEqual(result.between_seed_variance, 0)
        self.assertLess(result.data_main_effect_variance, 0)
        self.assertIsNotNone(result.algorithmic_variance_share)
        self.assertIsNone(result.total_order_algorithmic_variance_share)
        report = format_s5_report(
            result, estimand="test", computational_details={}
        )
        self.assertIn("total-order share would lie outside [0, 1]", report)

    def test_invalid_matrices_are_rejected(self) -> None:
        for values in (
            np.ones(4),
            np.ones((1, 3)),
            np.array([[1.0, np.nan], [2.0, 3.0]]),
        ):
            with self.subTest(shape=values.shape):
                with self.assertRaises(ValueError):
                    crossed_s5_diagnostics(values)


class ReproducibilityHelperTests(unittest.TestCase):
    def test_bootstrap_block_accepts_array_and_is_reproducible(self) -> None:
        seeds = np.array([11, 22], dtype=np.uint64)
        first = bootstrap_index_block(7, seeds)
        second = bootstrap_index_block(7, [11, 22])
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (2, 7))

    def test_seed_components_use_disjoint_stage_major_blocks(self) -> None:
        result = seed_component_blocks(
            list(range(20)),
            n_vectors=3,
            component_names=("folding", "modeling"),
            offset=2,
        )
        np.testing.assert_array_equal(result["folding"], [2, 3, 4])
        np.testing.assert_array_equal(result["modeling"], [5, 6, 7])

    def test_partitioned_bootstraps_use_reproducible_child_streams(self) -> None:
        first = partitioned_bootstrap_index_blocks(
            {"train": 7, "test": 4}, [11, 22]
        )
        second = partitioned_bootstrap_index_blocks(
            {"train": 7, "test": 4}, np.array([11, 22], dtype=np.uint64)
        )
        self.assertEqual(first["train"].shape, (2, 7))
        self.assertEqual(first["test"].shape, (2, 4))
        np.testing.assert_array_equal(first["train"], second["train"])
        np.testing.assert_array_equal(first["test"], second["test"])
        self.assertFalse(np.array_equal(first["train"][:, :4], first["test"]))

    def test_observed_summary_uses_sample_sd(self) -> None:
        result = observed_seed_summary([1.0, 2.0, 3.0])
        self.assertEqual(result.observed_data_seed_average, 2.0)
        self.assertEqual(result.observed_between_seed_sd, 1.0)

    def test_parallel_helpers_balance_work_and_resolve_auto_values(self) -> None:
        self.assertGreaterEqual(resolve_worker_count(-1), 1)
        self.assertEqual(resolve_worker_count(4, n_tasks=2), 2)
        self.assertEqual(resolve_batch_size(10, batch_size=0, n_jobs=3), 3)
        self.assertEqual(resolve_batch_size(10, batch_size=4, n_jobs=3), 4)
        chunks = parallel_chunk_ranges(10, n_jobs=2, tasks_per_worker=2)
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], 10)
        self.assertEqual(
            [index for start, stop in chunks for index in range(start, stop)],
            list(range(10)),
        )

    def test_log_contract_requires_all_six_items_per_estimand(self) -> None:
        report = format_s5_report(
            crossed_s5_diagnostics(np.arange(6, dtype=float).reshape(2, 3)),
            estimand="metric",
            computational_details={},
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.log"
            path.write_text(report, encoding="utf-8")
            validate_s5_log(path, ("metric",))
            path.write_text(report.replace("6. Computational details", ""),
                            encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_s5_log(path, ("metric",))


if __name__ == "__main__":
    unittest.main()
