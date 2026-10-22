"""Tests CPU sur des mesures synthétiques, sans modèle de diffusion."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from analyze_results import main, read_results, summarize


def result(result_id, method="base", prompt="a cat", seed=0, reward=0.0, beta=1.0, **values):
    row = {
        "config_id": "one-campaign", "result_id": result_id, "prompt": prompt,
        "seed": seed, "method": method, "reward_name": "luminance", "split": "dev",
        "reward_parameters_json": '{"target":0.7}', "beta": 0.0 if method == "base" else beta,
        "reward_png": reward, "reward_tensor": reward, "clip_score": None, "hpsv2": None,
        "seconds": 2.0, "peak_allocated_mib": None, "bank_id": "", "bank_score_key": "",
        "bank_generation_seconds": None, "bank_scoring_seconds": None,
    }
    row.update(values)
    return row


class AnalysisTests(unittest.TestCase):
    def test_pairing_uses_prompt_seed_reward_and_split(self):
        rows = [
            result("b0", reward=-3), result("b1", seed=1, reward=-1),
            result("c0", "cov", reward=-1), result("c1", "cov", seed=1, reward=-2),
            result("b_other", prompt="a dog", reward=0),
            result("c_other", "cov", prompt="a dog", reward=3),
        ]
        summaries, paired, totals = summarize(rows)
        cov = next(row for row in summaries if row["method"] == "cov")
        self.assertEqual([row["reward_gain"] for row in paired if row["method"] == "cov"], [2, -1, 3])
        self.assertAlmostEqual(cov["reward_gain_mean"], 4 / 3)
        self.assertAlmostEqual(cov["fraction_improved"], 2 / 3)
        self.assertEqual(cov["prompt_mean_gain_mean"], 1.75)
        self.assertEqual(cov["prompt_mean_gain_n"], 2)
        self.assertEqual(totals["missing_pairs"], [])

    def test_missing_metrics_and_base_are_not_zero(self):
        summaries, paired, totals = summarize([result("c", "cov", reward=4)])
        self.assertIsNone(paired[0]["reward_gain"])
        self.assertIsNone(summaries[0]["clip_score_mean"])
        self.assertIsNone(summaries[0]["reward_png_std"])
        self.assertEqual(summaries[0]["clip_score_n"], 0)
        self.assertEqual(summaries[0]["n_unpaired"], 1)
        self.assertEqual(len(totals["missing_pairs"]), 1)

    def test_betas_are_not_pooled(self):
        summaries, _, _ = summarize([result("b"), result("c1", "cov", beta=1), result("c2", "cov", beta=2)])
        self.assertEqual(len(summaries), 3)
        self.assertEqual({row["beta"] for row in summaries if row["method"] == "cov"}, {1, 2})

    def test_bank_generation_and_scoring_have_distinct_denominators(self):
        shared = {"bank_id": "bank", "bank_generation_seconds": 24.0, "bank_scoring_seconds": 6.0, "bank_score_key": "luminance"}
        rows = [result("b"), result("l1", "lidar", **shared), result("l2", "lidar", beta=2, **shared)]
        # Même banque, nouveau reward : génération partagée à trois, scoring à deux ou un.
        rows += [result("b", reward_name="symmetry", reward_parameters_json="{}")]
        rows += [result("l3", "lidar", reward_name="symmetry", reward_parameters_json="{}", **{**shared, "bank_score_key": "symmetry", "bank_scoring_seconds": 3.0})]
        _, paired, totals = summarize(rows)
        lidar = [row for row in paired if row["method"] == "lidar"]
        self.assertEqual([row["amortized_seconds"] for row in lidar], [13, 13, 13])
        self.assertEqual([row["standalone_seconds"] for row in lidar], [32, 32, 29])
        self.assertEqual([row["bank_generation_users"] for row in lidar], [3, 3, 3])
        self.assertEqual([row["bank_scoring_users"] for row in lidar], [2, 2, 1])
        self.assertEqual(totals["n_unique_generations"], 4)
        self.assertEqual(totals["generation_seconds_unique"], 8)
        self.assertEqual(totals["bank_generation_seconds_unique"], 24)
        self.assertEqual(totals["bank_scoring_seconds_unique"], 9)
        self.assertEqual(sum(row["bank_share_seconds"] for row in lidar), 33)

    def test_ambiguous_baseline_duplicate_and_mixed_campaign_rejected(self):
        for rows in (
            [result("b"), result("b")],
            [result("b"), result("b2")],
            [result("b"), result("c", "cov", config_id="other")],
        ):
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    summarize(rows)

    def test_conflicting_and_missing_bank_costs(self):
        shared = {"bank_id": "b", "bank_score_key": "r", "bank_generation_seconds": 4.0, "bank_scoring_seconds": 1.0}
        with self.assertRaises(ValueError):
            summarize([result("l1", "lidar", **shared), result("l2", "lidar", **{**shared, "bank_generation_seconds": 5.0})])
        _, paired, totals = summarize([result("l", "lidar", **{**shared, "bank_generation_seconds": None})])
        self.assertIsNone(paired[0]["amortized_seconds"])
        self.assertIsNone(totals["bank_generation_seconds_unique"])

    def test_csv_cli_preserves_missing_metrics_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "evaluation.csv"
            rows = [result("b"), result("c", "cov", reward=1)]
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            self.assertIsNone(read_results(csv_path)[0]["clip_score"])
            self.assertEqual(main(["--input", str(csv_path), "--output", str(root / "analysis"), "--no-plots"]), 0)
            for filename in ("summary.csv", "paired_results.csv", "analysis_metadata.json", "summary.md"):
                self.assertTrue((root / "analysis" / filename).is_file())
            metadata = json.loads((root / "analysis/analysis_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["n_unique_generations"], 2)
            csv_path.write_text(csv_path.read_text(encoding="utf-8").replace("a cat", "a dog", 1), encoding="utf-8")
            self.assertEqual(main(["--input", str(csv_path), "--output", str(root / "unpaired"), "--no-plots"]), 2)

    def test_nonfinite_csv_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            row = result("b", reward=float("nan"))
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=row)
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "non finie"):
                read_results(path)


if __name__ == "__main__":
    unittest.main()
