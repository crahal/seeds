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
    make_seed_plan,
    observed_rf_seed_grid,
)


class HousingSeedPlanTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
