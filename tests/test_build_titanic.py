"""Focused structural and preprocessing tests for the Titanic analysis."""

from __future__ import annotations

import sys
import tempfile
import unittest
import math
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "src" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from build_titanic import (  # noqa: E402
    Settings,
    _entropy_weight,
    evaluate_logistic,
    evaluate_models,
    evaluate_sgd,
    main,
    make_seed_plan,
    observed_sgd_seed_grid,
    wrangle_titanic,
)


class TitanicSeedPlanTests(unittest.TestCase):
    def test_defaults_use_all_cpus_and_automatic_checkpoint_batches(self) -> None:
        settings = Settings()
        self.assertEqual(settings.n_jobs, -1)
        self.assertEqual(settings.bootstrap_batch_size, 0)

    def test_default_visualization_grid_is_40_by_25_factorial(self) -> None:
        settings = Settings()
        plan = make_seed_plan(list(range(3_000)), settings)
        folding, modeling, n_folding, n_modeling = observed_sgd_seed_grid(
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


class TitanicWranglingTests(unittest.TestCase):
    def test_bootstrap_sample_tolerates_absent_groups_and_constant_fare(self) -> None:
        raw = pd.DataFrame(
            {
                "PassengerId": [1, 2, 3],
                "Survived": [0, 1, 0],
                "Pclass": [1, 2, 3],
                "Name": [
                    "Example, Mr. One",
                    "Example, Mrs. Two",
                    "Example, Master. Three",
                ],
                "Sex": ["male", "female", "male"],
                "Age": [np.nan, 40.0, 12.0],
                "SibSp": [0, 0, 1],
                "Parch": [0, 0, 1],
                "Ticket": ["A", "B", "C"],
                "Fare": [10.0, 10.0, 30.0],
                "Cabin": [None, None, "C1"],
                "Embarked": ["S", "S", "C"],
            }
        )
        # A bootstrap can omit most Sex x Pclass strata, leave one represented
        # stratum with no observed age, and make a formerly varying Fare constant.
        sampled = raw.iloc[[0, 0, 1, 1]]

        features, target = wrangle_titanic(sampled)

        self.assertEqual(len(features), len(sampled))
        np.testing.assert_array_equal(target.to_numpy(), [0, 0, 1, 1])
        self.assertTrue(all(np.issubdtype(dtype, np.number) for dtype in features.dtypes))
        self.assertTrue(np.isfinite(features.to_numpy(dtype=float)).all())


class TitanicEvaluationTests(unittest.TestCase):
    def test_entropy_weight_solves_the_bounded_upper_branch(self) -> None:
        for likelihood in (0.55, 0.7, 0.9, 0.99):
            probability = _entropy_weight(likelihood)
            recovered = math.exp(
                probability * math.log(probability)
                + (1 - probability) * math.log1p(-probability)
            )
            self.assertAlmostEqual(recovered, likelihood, places=11)
        self.assertEqual(_entropy_weight(0.4), 0.5)
        self.assertEqual(_entropy_weight(1.0), 0.999)

    def test_combined_evaluator_matches_standalone_model_evaluators(self) -> None:
        rng = np.random.default_rng(2025)
        features = rng.normal(size=(24, 5))
        target = np.tile([0, 1], 12)
        settings = Settings(n_folds=3, sgd_max_iter=100)

        combined = evaluate_models(
            features,
            target,
            folding_seed=17,
            modeling_seed=29,
            settings=settings,
        )
        logistic = evaluate_logistic(
            features, target, folding_seed=17, settings=settings
        )
        sgd = evaluate_sgd(
            features,
            target,
            folding_seed=17,
            modeling_seed=29,
            settings=settings,
        )

        self.assertEqual(combined.logistic, logistic)
        self.assertEqual(combined.sgd, sgd)
        for result in (logistic, sgd):
            self.assertTrue(np.isfinite([result.r2, result.imv, result.accuracy]).all())
            self.assertGreaterEqual(result.accuracy, 0.0)
            self.assertLessEqual(result.accuracy, 1.0)


class TitanicCliTests(unittest.TestCase):
    def test_tiny_end_to_end_run_writes_crossed_and_observed_outputs(self) -> None:
        row_number = np.arange(30)
        raw = pd.DataFrame(
            {
                "PassengerId": row_number + 1,
                "Survived": np.tile([0, 1, 0, 0, 1, 1], 5),
                "Pclass": np.tile([1, 2, 3, 2, 1], 6),
                "Name": [f"Example, Mr. Passenger{number}" for number in row_number],
                "Sex": np.tile(
                    ["male", "female", "female", "male", "male", "female"], 5
                ),
                "Age": 18 + (row_number * 7) % 50,
                "SibSp": row_number % 2,
                "Parch": (row_number // 2) % 2,
                "Ticket": [f"T{number}" for number in row_number],
                "Fare": 5.0 + (row_number * 11) % 50,
                "Cabin": [None] * len(row_number),
                "Embarked": np.tile(["S", "C", "Q"], 10),
            }
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "train.csv"
            seed_path = root / "seeds.txt"
            output_dir = root / "output"
            log_path = root / "run.log"
            accuracy_path = root / "accuracy.txt"
            raw.to_csv(csv_path, index=False)
            seed_path.write_text(
                "".join(f"{seed}\n" for seed in range(100, 120)),
                encoding="utf-8",
            )

            main(
                [
                    "--titanic-csv",
                    str(csv_path),
                    "--seed-list",
                    str(seed_path),
                    "--output-dir",
                    str(output_dir),
                    "--log-path",
                    str(log_path),
                    "--accuracy-path",
                    str(accuracy_path),
                    "--n-bootstrap-resamples",
                    "2",
                    "--n-seed-vectors",
                    "2",
                    "--n-visualization-runs",
                    "4",
                    "--n-jobs",
                    "1",
                    "--bootstrap-batch-size",
                    "1",
                    "--n-folds",
                    "2",
                    "--sgd-max-iter",
                    "50",
                    "--logistic-max-iter",
                    "25",
                ]
            )

            crossed = pd.read_csv(
                output_dir / "titanic_crossed_bootstrap_scores.csv"
            )
            observed_sgd = pd.read_csv(output_dir / "titanic_outputs_sgd.csv")
            self.assertEqual(len(crossed), 4)
            self.assertEqual(len(observed_sgd), 4)
            self.assertTrue((output_dir / "titanic_s5_diagnostics.json").is_file())
            self.assertEqual(
                np.load(output_dir / "titanic_bootstrap_indices.npy").shape,
                (2, len(raw)),
            )
            self.assertEqual(len(accuracy_path.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
