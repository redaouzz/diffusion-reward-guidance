"""Tests CPU sans poids téléchargés : PNG, contrat CSV, cache et séparation des métriques."""

from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import weakref

import numpy as np
from PIL import Image

import evaluate


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.image = self.root / "image.png"
        Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8)).save(self.image)

    def make_result(self, name="base", method="base", reward="base", bank_id=None):
        folder = self.root / "results" / name
        folder.mkdir(parents=True)
        image_path = folder / "image.png"
        image_path.write_bytes(self.image.read_bytes())
        record = {"status": "complete", "method": method, "prompt": "a red flower", "prompt_index": 4,
                  "seed": 0, "config_id": "campaign", "split": "dev", "image": "image.png",
                  "reward_name": reward, "reward_parameters": {"target": 0.7}, "beta": 10,
                  "reward_tensor": -0.1, "bank_id": bank_id, "bank_score_key": "lum",
                  "generation_cost": {"seconds": 5, "peak_allocated_mib": None},
                  "artifacts": {"image.png": evaluate.sha256_file(image_path)}}
        result_path = folder / "result.json"
        result_path.write_text(json.dumps(record), encoding="utf-8")
        return result_path

    def manifest(self):
        value = {"task_count": 4, "config": {"rewards": [{"name": "luminance", "parameters": {"target": 0.7}, "betas": [10]},
                                          {"name": "symmetry", "parameters": {}, "betas": [1]}]}}
        (self.root / "manifest.json").write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def mock_completed(path):
        # Mock du contrat de lecture uniquement : les tests du runner vérifient sa vraie reprise.
        record = json.loads(path.read_text(encoding="utf-8"))
        for filename, digest in record["artifacts"].items():
            if evaluate.sha256_file(path.parent / filename) != digest:
                return None
        return record

    def test_rewards_from_actual_png(self):
        expected = -(128 / 255 - 0.7) ** 2
        self.assertAlmostEqual(evaluate.reward_from_png(self.image, "", "luminance", {"target": 0.7}), expected, places=7)
        self.assertEqual(evaluate.reward_from_png(self.image, "", "symmetry", {}), 0)
        mask_reward = evaluate.reward_from_png(self.image, "", "spatial_mask", {"radius_fraction": 0.25})
        self.assertLess(mask_reward, 0)
        with self.assertRaises(ValueError):
            evaluate.reward_from_png(self.image, "", "missing_reward", {})

    def test_expand_base_and_preserve_bank_cost(self):
        self.manifest()
        self.make_result()
        self.make_result("lidar", "lidar", "luminance", "bank")
        bank = self.root / "banks" / "bank"
        (bank / "scores").mkdir(parents=True)
        (bank / "bank.json").write_text(json.dumps({"generation_cost": {"seconds": 20}, "artifacts": {}}), encoding="utf-8")
        (bank / "scores" / "lum.json").write_text(json.dumps({"scoring_cost": {"seconds": 3}, "artifacts": {},
                       "bank_sha256": evaluate.sha256_file(bank / "bank.json")}), encoding="utf-8")
        with patch.dict(sys.modules, {"run_experiment": types.SimpleNamespace(completed_result=self.mock_completed)}):
            rows = evaluate.collect_rows(self.root)
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["reward_name"] for row in rows[:2]], ["luminance", "symmetry"])
        self.assertIsNone(rows[0]["reward_tensor"])
        self.assertEqual(rows[2]["bank_generation_seconds"], 20)
        self.assertEqual(rows[2]["bank_scoring_seconds"], 3)
        self.assertEqual(rows[2]["seconds"], 5)

    def test_corrupt_image_excluded(self):
        self.manifest()
        result_path = self.make_result()
        (result_path.parent / "image.png").write_bytes(b"partial PNG")
        with patch.dict(sys.modules, {"run_experiment": types.SimpleNamespace(completed_result=self.mock_completed)}):
            self.assertEqual(evaluate.collect_rows(self.root), [])

    def test_cache_reuse_and_content_invalidation(self):
        row = {"image_path": str(self.image), "prompt": "x", "result_id": "id", "reward_name": "symmetry",
               "reward_parameters_json": "{}"}
        calls = []
        def score(record):
            calls.append(record["result_id"])
            return 0.25
        evaluate.score_pass([row], "clip", score, self.root, {"model": "mock"})
        evaluate.score_pass([row], "clip", score, self.root, {"model": "mock"})
        self.assertEqual(len(calls), 1)
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(self.image)
        evaluate.score_pass([row], "clip", score, self.root, {"model": "mock"})
        self.assertEqual(len(calls), 2)
        evaluate.score_pass([row], "clip", score, self.root, {"model": "mock-v2"})
        self.assertEqual(len(calls), 3)

    def test_nonfinite_score_never_committed(self):
        row = {"image_path": str(self.image), "prompt": "x", "result_id": "id"}
        with self.assertRaises(ValueError):
            evaluate.score_pass([row], "hps", lambda _: float("nan"), self.root, {})
        self.assertEqual(list((self.root / "evaluation_cache" / "hps").glob("*.json")), [])

    def test_interrupted_score_pass_resumes(self):
        rows = [{"image_path": str(self.image), "prompt": prompt, "result_id": prompt} for prompt in ("a", "b")]
        def interrupted(row):
            if row["prompt"] == "b":
                raise RuntimeError("simulated interruption")
            return 1.0
        with self.assertRaises(RuntimeError):
            evaluate.score_pass(rows, "clip", interrupted, self.root, {})
        calls = []
        evaluate.score_pass(rows, "clip", lambda row: calls.append(row["prompt"]) or 2.0, self.root, {})
        self.assertEqual(calls, ["b"])
        self.assertEqual([row["clip_score"] for row in rows], [1, 2])

    def test_end_to_end_cpu_rewards_no_metric_model(self):
        self.manifest()
        self.make_result()
        with patch.dict(sys.modules, {"run_experiment": types.SimpleNamespace(completed_result=self.mock_completed)}), \
                patch.object(evaluate, "load_clip_scorer", side_effect=AssertionError("unexpected model")), \
                patch.object(evaluate, "load_hps_scorer", side_effect=AssertionError("unexpected model")):
            rows = evaluate.evaluate(self.root, metrics=["reward"], device="cpu")
        self.assertEqual(len(rows), 2)
        self.assertTrue((self.root / "evaluation.csv").exists())
        self.assertIsNone(rows[0]["clip_score"])
        self.assertIsNone(rows[0]["peak_allocated_mib"])
        provenance = json.loads((self.root / "evaluation_provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["complete_metrics"], ["reward"])

    def test_generation_coverage_counts_images_not_expanded_rows(self):
        self.manifest()
        self.make_result()
        (self.root / "failures").mkdir()
        (self.root / "failures" / "old_attempt.json").write_text("{}", encoding="utf-8")
        with patch.dict(sys.modules, {"run_experiment": types.SimpleNamespace(completed_result=self.mock_completed)}):
            rows = evaluate.evaluate(self.root, metrics=["reward"], device="cpu")
        provenance = json.loads((self.root / "evaluation_provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 2)  # Une BASE est développée pour deux récompenses.
        self.assertEqual(provenance["planned_generations"], 4)
        self.assertEqual(provenance["completed_generations"], 1)
        self.assertEqual(provenance["missing_generations"], 3)
        self.assertEqual(provenance["failure_records_count"], 1)
        self.assertFalse(provenance["generation_complete"])
        self.assertIn("not a final failure rate", provenance["coverage_note"])

    def test_metric_cache_identity_distinguishes_gpu_models(self):
        with patch.object(evaluate.torch.cuda, "get_device_name", side_effect=["NVIDIA T4", "NVIDIA H100 80GB"]):
            t4_spec = evaluate.metric_spec("clip", "cuda")
            h100_spec = evaluate.metric_spec("clip", "cuda")
        self.assertNotEqual(t4_spec, h100_spec)
        self.assertIn("cuda", t4_spec)

    def test_metric_models_released_and_partial_pass_preserves_reward(self):
        self.manifest()
        self.make_result()
        references = []
        def loader(device, cache):
            self.assertTrue(all(reference() is None for reference in references))
            model = evaluate.torch.nn.Linear(1, 1)
            references.append(weakref.ref(model))
            def score(path, prompt):
                return float(model.weight.numel())
            return score
        with patch.dict(sys.modules, {"run_experiment": types.SimpleNamespace(completed_result=self.mock_completed)}), \
                patch.object(evaluate, "load_clip_scorer", side_effect=loader), \
                patch.object(evaluate, "load_hps_scorer", side_effect=loader):
            previous = evaluate.evaluate(self.root, metrics=["reward"], device="cpu")
            rows = evaluate.evaluate(self.root, metrics=["clip", "hps"], device="cpu")
        self.assertEqual([r["reward_png"] for r in rows], [r["reward_png"] for r in previous])
        self.assertEqual([r["clip_score"] for r in rows], [1, 1])
        self.assertEqual([r["hpsv2"] for r in rows], [1, 1])
        self.assertTrue(all(reference() is None for reference in references))

    def test_actual_runner_files_through_evaluation_and_analysis(self):
        # Seuls les modèles de génération sont simulés; fichiers, récompensesPNG et CSV sont réels.
        from test_experiment import ExperimentTests
        import run_experiment
        import analyze_results
        fixture = ExperimentTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture.config_data["seeds"] = [0]
        fixture.config_data["rewards"][0]["betas"] = [1]
        with ExitStack() as stack:
            fixture.mock_models(stack)
            run_experiment.run_campaign(fixture.config(), fixture.output, "cpu")
        rows = evaluate.evaluate(fixture.output, metrics=["reward"], device="cpu")
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["reward_png"] is not None for r in rows))
        base = next(row for row in rows if row["method"] == "base")
        self.assertIsNotNone(base["reward_tensor"])
        self.assertTrue(all(r["peak_allocated_mib"] is None for r in rows))
        provenance = json.loads((fixture.output / "evaluation_provenance.json").read_text(encoding="utf-8"))
        self.assertTrue(provenance["generation_complete"])
        self.assertEqual(provenance["missing_generations"], 0)
        analyze_results.main(["--input", str(fixture.output / "evaluation.csv"),
                              "--output", str(fixture.output / "analysis"), "--no-plots"])
        self.assertTrue((fixture.output / "analysis" / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
