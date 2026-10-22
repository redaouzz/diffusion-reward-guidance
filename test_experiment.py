"""Tests CPU de l'orchestration ; les samplers sont simulés, jamais SD1.5 validé ici."""

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import run_experiment as run
from test_baseline import make_test_scheduler


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "dev.txt").write_text("a sandwich\na red flower\n", encoding="utf-8")
        (self.root / "heldout.txt").write_text("a green bike\n", encoding="utf-8")
        self.config_data = dict(prompts_file="dev.txt", exclude_prompts_file="heldout.txt", split="dev",
                                prompt_limit=1, seeds=[0, 1], methods=list(run.METHODS),
                                rewards=[dict(name="luminance", betas=[1, 2])],
                                bank=dict(size=2, num_steps=5, seed_start=10000))
        self.output = self.root / "results"

    def tearDown(self):
        self.temporary.cleanup()

    def config(self):
        run.atomic_json(self.root / "config.json", self.config_data)
        return run.load_config(self.root / "config.json")

    def mock_models(self, stack):
        vae = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        vae.config = SimpleNamespace(scaling_factor=0.18215)
        pipe = SimpleNamespace(vae=vae, unet=torch.nn.Linear(1, 1),
                               scheduler=make_test_scheduler(), _execution_device=torch.device("cpu"))
        mocks = {}
        mocks["load"] = stack.enter_context(patch.object(run, "load_pipeline", return_value=pipe))
        stack.enter_context(patch.object(run, "encode_prompt_cfg", return_value=(torch.zeros(1), torch.zeros(1))))
        stack.enter_context(patch.object(run, "make_initial_noise", side_effect=lambda p, seed:
                                        torch.full((1, 4, 64, 64), seed * 1e-5)))
        mocks["base"] = stack.enter_context(patch.object(run, "sample_ddim", side_effect=lambda p, noise, *a, **kw:
                                                        (noise + 0.1, [])))
        for method in ("cov", "full", "lidar"):
            def guided(p, noise, *args, diagnostics, **kwargs):
                diagnostics.append(dict(status="ok", relative_noise_correction=0.01, ess_reward=1.5))
                return noise + 0.2, [], diagnostics
            mocks[method] = stack.enter_context(patch.object(run, "sample_" + method, side_effect=guided))
        stack.enter_context(patch.object(run, "decode_latents_differentiable",
                                        side_effect=lambda vae, z: z.mean().expand(1, 3, 512, 512).clamp(0, 1)))
        return mocks

    def test_matrix_unique_baseline_shared_and_betas_shared(self):
        self.config_data["rewards"].append(dict(name="symmetry", betas=[1, 2]))
        tasks = run.make_tasks(self.config())
        self.assertEqual(len(tasks), 26)  # 2 seeds * (1 BASE + 2 rewards * 2 beta * 3 methods).
        self.assertEqual(sum(t["method"] == "base" for t in tasks), 2)
        self.assertEqual(len({t["result_id"] for t in tasks}), len(tasks))
        self.assertEqual(len({t["config_id"] for t in tasks}), 1)
        for method in ("cov", "full", "lidar"):
            self.assertEqual({t["beta"] for t in tasks if t["method"] == method}, {1, 2})

    def test_reject_overlap_invalid_beta_seed_and_unknown_parameters(self):
        original = json.loads(json.dumps(self.config_data))
        for change in ({"exclude_prompts_file": "dev.txt"}, {"seeds": [10000]},
                       {"seeds": [0, 0]}, {"height": 256}, {"prompt_limit": 10},
                       {"rewards": [dict(name="luminance", betas=None)]},
                       {"rewards": [dict(name="luminance", betas=[float("inf")])]},
                       {"rewards": [dict(name="luminance", betas=[1], parameters={"scale": 2})]}):
            self.config_data = original | change
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.config()

    def test_prompt_reading_and_hash_changes(self):
        first = run.make_tasks(self.config())[0]["config_id"]
        (self.root / "dev.txt").write_text("a changed sandwich\na red flower\n", encoding="utf-8")
        self.assertNotEqual(first, run.make_tasks(self.config())[0]["config_id"])
        (self.root / "dev.txt").write_text("duplicate\nduplicate\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.config()

    def test_known_heldout_cannot_be_renamed_into_dev(self):
        heldout_prompt = run.read_prompts(run.ROOT / "data" / "prompts_50_heldout.txt")[0]
        (self.root / "dev.txt").write_text(heldout_prompt + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Inversion"):
            self.config()

    def test_imagereward_loaded_only_once(self):
        model = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        self.output.mkdir()
        with patch.object(run, "load_imagereward", return_value=model) as loader:
            first = run.place_imagereward(None, "cpu", self.output)
            second = run.place_imagereward(first, "cpu", self.output)
            self.assertIs(first, second)
            loader.assert_called_once_with("cpu")

    def test_dry_run_without_weights_or_output(self):
        with patch.object(run, "load_pipeline") as loader:
            report = run.run_campaign(self.config(), self.output, "cpu", dry_run=True)
        self.assertEqual(report["task_count"], 14)
        self.assertFalse(self.output.exists())
        loader.assert_not_called()

    def test_artifacts_partial_corrupt_and_successful_commit(self):
        image = torch.zeros(1, 3, 8, 8)
        with patch.object(run, "atomic_json", side_effect=OSError("interrupted before manifest")):
            with self.assertRaises(OSError):
                run.commit_files(self.output, "result.json", {}, image=image)
        self.assertTrue((self.output / "image.png").exists())
        self.assertIsNone(run.completed_result(self.output / "result.json"))
        run.commit_files(self.output, "result.json", {}, image=image)
        self.assertIsNotNone(run.completed_result(self.output / "result.json"))
        (self.output / "image.png").write_bytes(b"incomplete PNG")
        self.assertIsNone(run.completed_result(self.output / "result.json"))

    def test_single_writer_lock(self):
        with run.campaign_lock(self.output):
            with self.assertRaises(RuntimeError):
                with run.campaign_lock(self.output):
                    pass
        self.assertFalse((self.output / ".lock").exists())

    def test_run_bank_reuse_and_no_work_after_completion(self):
        with ExitStack() as stack:
            mocks = self.mock_models(stack)
            result = run.run_campaign(self.config(), self.output, "cpu")
            self.assertEqual(result["generated"], 14)
            self.assertEqual(mocks["base"].call_count, 4)  # BASE x2 + bank x2, shared 4 LiDAR trajectories.
            self.assertEqual(mocks["lidar"].call_count, 4)
            mocks["load"].reset_mock()
            repeat = run.run_campaign(self.config(), self.output, "cpu")
            self.assertEqual(repeat["generated"], 0)
            mocks["load"].assert_not_called()
        self.assertEqual(len(list((self.output / "banks").glob("*/bank.json"))), 1)
        self.assertEqual(len(list((self.output / "banks").glob("*/scores/*.json"))), 1)
        records = [run.completed_result(p) for p in (self.output / "results").glob("*/result.json")]
        self.assertTrue(all(records))
        self.assertTrue(all(r["runtime_sha256"] == run.sha256(self.output / "runtime.json") for r in records))
        self.assertTrue(all(r["baseline_rewards"] for r in records if r["method"] == "base"))

    def test_interrupted_run_preserves_completed_results_and_resumes(self):
        with ExitStack() as stack:
            mocks = self.mock_models(stack)
            guided = mocks["full"].side_effect
            mocks["full"].side_effect = RuntimeError("simulated interruption")
            with self.assertRaises(RuntimeError):
                run.run_campaign(self.config(), self.output, "cpu")
            completed = {p: run.sha256(p) for p in (self.output / "results").glob("*/result.json")}
            self.assertEqual(len(completed), 2)  # BASE then COV ; FULL failed visibly.
            self.assertEqual(len(list((self.output / "failures").glob("*.json"))), 1)
            mocks["full"].side_effect = guided
            result = run.run_campaign(self.config(), self.output, "cpu")
            self.assertEqual(result["generated"], 12)
            self.assertTrue(all(run.sha256(p) == old_hash for p, old_hash in completed.items()))
            self.config_data["seeds"] = [0]
            with self.assertRaises(ValueError):
                run.run_campaign(self.config(), self.output, "cpu")

    def test_bank_scoring_recomputed_if_bank_changes(self):
        directory = self.root / "bank"
        run.atomic_json(directory / "bank.json", {"bank": 1})
        pipe = SimpleNamespace(vae=None, _execution_device="cpu")
        bank = torch.zeros(2, 4, 2, 2)
        spec = dict(name="luminance", parameters={"target": 0.7})
        reward_fn = lambda images: images.mean((1, 2, 3))
        with patch.object(run, "decode_latents_differentiable", return_value=torch.ones(1, 3, 2, 2)) as decode:
            run.score_bank(pipe, bank, reward_fn, directory, spec)
            run.score_bank(pipe, bank, reward_fn, directory, spec)
            self.assertEqual(decode.call_count, 2)
            run.atomic_json(directory / "bank.json", {"bank": 2})
            run.score_bank(pipe, bank, reward_fn, directory, spec)
            self.assertEqual(decode.call_count, 4)

    def test_corrupt_bank_invalidates_lidar_and_rebuilds_only_its_dependents(self):
        self.config_data["seeds"] = [0]
        self.config_data["rewards"][0]["betas"] = [1]
        with ExitStack() as stack:
            mocks = self.mock_models(stack)
            run.run_campaign(self.config(), self.output, "cpu")
            lidar_task = next(t for t in run.make_tasks(self.config()) if t["method"] == "lidar")
            result_path = self.output / "results" / lidar_task["result_id"] / "result.json"
            self.assertIsNotNone(run.completed_result(result_path))
            bank_tensor = next((self.output / "banks").glob("*/candidate_000/latent.pt"))
            bank_tensor.write_bytes(b"interrupted/corrupt tensor")
            self.assertIsNone(run.completed_result(result_path))
            mocks["base"].reset_mock()
            report = run.run_campaign(self.config(), self.output, "cpu")
            self.assertEqual(report["generated"], 1)
            self.assertEqual(mocks["base"].call_count, 1)  # One corrupt bank candidate, BASE preserved.
            self.assertIsNotNone(run.completed_result(result_path))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
