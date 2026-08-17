"""Focused design, compatibility, and tiny-run tests for COMPAS."""

from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "src" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

import build_compass as compass  # noqa: E402
from build_compass import (  # noqa: E402
    SCORE_KEYS,
    Settings,
    _evaluate_bootstrap_row,
    main,
    make_seed_plan,
    per_uid_stats,
    run_oof_one_seed,
)


def _tiny_features_and_target(n_rows: int = 24) -> tuple[np.ndarray, np.ndarray]:
    target = np.tile([0, 1], n_rows // 2)
    features = np.column_stack(
        (
            18 + 20 * target + np.arange(n_rows) % 5,
            np.arange(n_rows) % 7,
        )
    ).astype(float)
    return features, target


def _write_compas_csv(path: Path, n_rows: int = 48) -> pd.DataFrame:
    row = np.arange(n_rows)
    target = np.tile([0, 1], n_rows // 2)
    frame = pd.DataFrame(
        {
            "id": row + 1,
            "days_b_screening_arrest": (row % 21) - 10,
            "is_recid": target,
            "c_charge_degree": np.where(row % 3, "F", "M"),
            "score_text": np.where(target, "High", "Low"),
            "age": 18 + 20 * target + row % 5,
            "priors_count": row % 7,
            "two_year_recid": target,
            "decile_score": np.where(target, 8, 3),
        }
    )
    frame.to_csv(path, index=False)
    return frame


class CompassDesignTests(unittest.TestCase):
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
        settings.validate()

        plan = make_seed_plan(list(range(1_100)), settings)
        np.testing.assert_array_equal(plan.joint, np.arange(1_000))
        np.testing.assert_array_equal(plan.bootstrap, np.arange(1_000, 1_100))
        self.assertTrue(set(plan.joint).isdisjoint(set(plan.bootstrap)))

    def test_bootstrap_row_is_reused_for_every_joint_seed(self) -> None:
        features, target = _tiny_features_and_target(8)
        indices = np.array(
            [[0, 1, 2, 3, 4, 5, 6, 7], [7, 6, 6, 5, 4, 3, 2, 1]],
            dtype=np.int64,
        )
        settings = Settings(
            n_bootstrap_resamples=2,
            n_seed_vectors=2,
            n_visualization_runs=2,
            n_jobs=1,
        )
        seen: list[tuple[int, np.ndarray, np.ndarray]] = []

        def fake_oof(
            sampled_features: np.ndarray,
            sampled_target: np.ndarray,
            seed: int,
            _settings: Settings,
        ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
            seen.append((seed, sampled_features.copy(), sampled_target.copy()))
            probabilities = 0.2 + 0.6 * sampled_target
            return {
                model: probabilities.copy() for model in compass.MODEL_KEYS
            }, {"convergence_warnings": 0, "other_warnings": 0}

        with patch.object(compass, "_run_oof_audited", side_effect=fake_oof):
            row, scores, issues = _evaluate_bootstrap_row(
                1,
                indices,
                features,
                target,
                np.array([101, 202], dtype=np.uint64),
                settings,
            )

        self.assertEqual(row, 1)
        self.assertEqual([item[0] for item in seen], [101, 202])
        for _seed, sampled_features, sampled_target in seen:
            np.testing.assert_array_equal(sampled_features, features[indices[1]])
            np.testing.assert_array_equal(sampled_target, target[indices[1]])
        self.assertEqual(set(scores), set(SCORE_KEYS))
        for values in scores.values():
            np.testing.assert_array_equal(values, np.ones(2))
        self.assertEqual(issues, {"convergence_warnings": 0, "other_warnings": 0})


class CompassCompatibilityTests(unittest.TestCase):
    def test_oof_return_order_and_seeded_determinism(self) -> None:
        features, target = _tiny_features_and_target()
        kwargs = {
            "rf_n_estimators": 3,
            "rf_max_depth": 2,
            "model_max_iter": 80,
            "mlp_hidden_units": 3,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first = run_oof_one_seed(features, target, 17, 2, **kwargs)
            repeated = run_oof_one_seed(features, target, 17, 2, **kwargs)

        self.assertEqual(len(first), 3)
        rf, lr, nn = first
        for actual, again in zip((rf, lr, nn), repeated, strict=True):
            self.assertEqual(actual.shape, (len(target),))
            self.assertTrue(np.isfinite(actual).all())
            self.assertTrue(((0 <= actual) & (actual <= 1)).all())
            np.testing.assert_array_equal(actual, again)

        # The public legacy order is RF, logistic regression, then neural net.
        self.assertFalse(np.array_equal(rf, lr))
        self.assertFalse(np.array_equal(lr, nn))

    def test_per_uid_stats_retains_legacy_seven_tuple(self) -> None:
        values = np.array(
            [[0.1, 0.2, 0.8, 0.9], [0.4, 0.5, 0.6, 0.7]], dtype=float
        )
        result = per_uid_stats(values, threshold=0.5)

        self.assertEqual(len(result), 7)
        mean, sd, q10, q50, q90, flips, flip_rate = result
        np.testing.assert_allclose(mean, values.mean(axis=1))
        np.testing.assert_allclose(sd, values.std(axis=1, ddof=1))
        np.testing.assert_allclose(q10, np.percentile(values, 10, axis=1))
        np.testing.assert_allclose(q50, np.percentile(values, 50, axis=1))
        np.testing.assert_allclose(q90, np.percentile(values, 90, axis=1))
        np.testing.assert_array_equal(flips, [2, 1])
        np.testing.assert_allclose(flip_rate, [0.5, 0.25])


class CompassCliTests(unittest.TestCase):
    def test_tiny_run_writes_s5_and_legacy_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "compas.csv"
            output_dir = root / "results"
            seed_path = root / "seeds.txt"
            log_path = root / "compass.log"
            source = _write_compas_csv(data_path)
            seed_path.write_text(
                "".join(f"{seed}\n" for seed in range(100, 120)),
                encoding="utf-8",
            )

            main(
                [
                    "--data-path",
                    str(data_path),
                    "--seed-list",
                    str(seed_path),
                    "--output-dir",
                    str(output_dir),
                    "--log-path",
                    str(log_path),
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
                    "--observed-batch-size",
                    "2",
                    "--rf-n-estimators",
                    "2",
                    "--rf-max-depth",
                    "2",
                    "--model-max-iter",
                    "60",
                    "--mlp-hidden-units",
                    "2",
                ]
            )

            crossed = pd.read_csv(
                output_dir / "compass_crossed_bootstrap_scores.csv"
            )
            self.assertEqual(crossed.shape[0], 4)
            self.assertEqual(
                list(crossed.columns),
                [
                    "Bootstrap_Replicate",
                    "Seed_Vector",
                    "Bootstrap_Seed",
                    "Joint_Folding_Model_Seed",
                    *SCORE_KEYS,
                ],
            )
            for replicate, group in crossed.groupby("Bootstrap_Replicate"):
                self.assertEqual(group["Bootstrap_Seed"].nunique(), 1)
                self.assertEqual(
                    group["Bootstrap_Seed"].iloc[0], 104 + replicate
                )
                self.assertEqual(group["Seed_Vector"].tolist(), [0, 1])
                self.assertEqual(
                    group["Joint_Folding_Model_Seed"].tolist(), [100, 101]
                )

            visualization = pd.read_csv(
                output_dir / "compass_visualization_runs.csv"
            )
            self.assertEqual(visualization.shape[0], 4)
            self.assertEqual(
                list(visualization.columns),
                ["Seed_Vector", "Joint_Folding_Model_Seed", *SCORE_KEYS],
            )

            prediction_path = (
                output_dir
                / "uid_oof_predictions_4seeds_rf_lr_nn_compas_2folds.csv"
            )
            summary_path = (
                output_dir
                / "uid_summary_instability_4seeds_2folds_rf_lr_nn.csv"
            )
            predictions = pd.read_csv(prediction_path)
            summary = pd.read_csv(summary_path)
            self.assertEqual(len(predictions), len(source))
            self.assertEqual(len(summary), len(source))
            self.assertEqual(predictions.columns[0], "UID")
            self.assertEqual(summary.columns[0], "UID")
            self.assertEqual(
                [column for column in predictions if column.startswith("y_hat_")],
                [
                    f"y_hat_{model}_seed{seed}"
                    for model in ("rf", "lr", "nn")
                    for seed in range(100, 104)
                ],
            )
            self.assertEqual(
                list(summary.columns),
                [
                    "UID",
                    "y",
                    "rf_mu",
                    "rf_sd",
                    "rf_fliprate",
                    "lr_mu",
                    "lr_sd",
                    "lr_fliprate",
                    "nn_mu",
                    "nn_sd",
                    "nn_fliprate",
                    "age",
                    "priors",
                    "compas_decile",
                ],
            )

            self.assertEqual(
                np.load(output_dir / "compass_bootstrap_indices.npy").shape,
                (2, len(source)),
            )
            self.assertTrue((output_dir / "compass_s5_diagnostics.csv").is_file())
            self.assertTrue((output_dir / "compass_s5_diagnostics.json").is_file())
            self.assertTrue((output_dir / "compass_score_provenance.json").is_file())

            log = log_path.read_text(encoding="utf-8")
            self.assertEqual(
                log.count("S5 recommended reporting and diagnostics:"),
                len(SCORE_KEYS),
            )
            for heading in (
                "1. Seed-averaged estimate",
                "2. Data uncertainty",
                "3. Bias-corrected between-seed variability",
                "4. Relative importance of algorithmic randomness",
                "5. Algorithmic variance share",
                "6. Computational details",
            ):
                self.assertIn(heading, log)


if __name__ == "__main__":
    unittest.main()
