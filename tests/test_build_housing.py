"""Small structural tests for the housing-specific seed plans."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "src" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from build_housing_seeds import (  # noqa: E402
    Settings,
    _split_dataset,
    make_seed_plan,
    observed_rf_seed_grid,
)
from sklearn.model_selection import train_test_split  # noqa: E402


class HousingSeedPlanTests(unittest.TestCase):
    def test_defaults_use_all_cpus_and_automatic_checkpoint_batches(self) -> None:
        settings = Settings()
        self.assertEqual(settings.n_jobs, -1)
        self.assertEqual(settings.bootstrap_batch_size, 0)

    def test_default_visualization_grid_is_40_by_25_factorial(self) -> None:
        settings = Settings()
        plan = make_seed_plan(list(range(3_000)), settings)
        folding, modeling, n_folding, n_modeling = observed_rf_seed_grid(
            plan, settings.n_visualization_runs
        )

        self.assertEqual((n_folding, n_modeling), (40, 25))
        self.assertEqual(len(folding), 1_000)
        self.assertEqual(len(modeling), 1_000)
        self.assertEqual(np.unique(folding).size, 40)
        self.assertEqual(np.unique(modeling).size, 25)
        self.assertEqual(len(set(zip(folding, modeling, strict=True))), 1_000)

    def test_seed_stages_and_bootstrap_block_are_disjoint(self) -> None:
        settings = Settings(
            n_bootstrap_resamples=3,
            n_seed_vectors=3,
            n_visualization_runs=4,
        )
        plan = make_seed_plan(list(range(20)), settings)
        self.assertTrue(set(plan.folding).isdisjoint(set(plan.modeling)))
        self.assertTrue(set(plan.folding).isdisjoint(set(plan.bootstrap)))
        self.assertTrue(set(plan.modeling).isdisjoint(set(plan.bootstrap)))

    def test_cached_positional_split_matches_legacy_value_split(self) -> None:
        features = np.arange(120, dtype=float).reshape(30, 4)
        target = np.arange(30, dtype=float)
        settings = Settings(test_size=0.3)
        actual = _split_dataset(
            features, target, folding_seed=1234, settings=settings
        )
        expected = train_test_split(
            features,
            target,
            test_size=settings.test_size,
            random_state=1234,
            shuffle=True,
        )
        for actual_array, expected_array in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(actual_array, expected_array)


if __name__ == "__main__":
    unittest.main()
