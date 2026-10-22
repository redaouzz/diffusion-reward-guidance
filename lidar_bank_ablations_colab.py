"""Deux ablations de banque LiDAR, depuis le notebook 05 déjà exécuté.

Le sampler et la récompense restent ceux du projet. Aucun réglage automatique.
"""

import csv
import hashlib
import importlib.metadata
import json
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image, ImageDraw

import baseline
import guidance_lidar
import rewards


BETA = 66.37033847314085


def save_json(path, value):
    """JSON strict : les NaN ne sont pas masqués dans les rapports."""
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def prepare_ablations(pipe, reference_dir):
    """Vérifie la session et copie la référence dans un nouveau dossier indépendant."""
    reference_dir = Path(reference_dir).resolve()
    env = json.loads((reference_dir / "environment.json").read_text(encoding="utf-8"))
    checks = json.loads((reference_dir / "algorithm_checks.json").read_text(encoding="utf-8"))
    assert checks["gpu_status"] == "passed_for_this_pilot", "Le pilote initial doit être terminé."
    assert env["reward"] == "luminance" and env["eta"] == 0
    assert env["bank_size"] == 8 and env["lookahead_steps"] == 5
    assert env["bank_seeds"] == list(range(10000, 10008))
    assert not set(range(10000, 10016)) & {env["seed"], 0, 1, 2, 3}
    assert env["unet_dtype"] == str(pipe.unet.dtype) and env["vae_dtype"] == str(pipe.vae.dtype)
    assert all(not m.training and all(not p.requires_grad and p.grad is None for p in m.parameters())
               for m in (pipe.unet, pipe.vae))
    public_config = {k: v for k, v in dict(pipe.scheduler.config).items() if not k.startswith("_")}
    assert public_config == {k: v for k, v in env["scheduler"].items() if not k.startswith("_")}
    for package, version in env["versions"].items():
        assert importlib.metadata.version(package) == version, f"Version modifiée : {package}"
    for module in (baseline, rewards, guidance_lidar):
        source = Path(module.__file__)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == env["source_sha256"][source.name]

    device = next(pipe.unet.parameters()).device
    inputs = torch.load(reference_dir / "inputs.pt", map_location="cpu", weights_only=True)
    noise = baseline.make_initial_noise(pipe, env["seed"], height=env["resolution"][0], width=env["resolution"][1])
    assert torch.equal(noise.cpu(), inputs["initial_noise"]), "Bruit initial incompatible."
    pipe.scheduler.set_timesteps(env["num_steps"], device=device)
    assert pipe.scheduler.timesteps.cpu().tolist() == env["timesteps"]
    metadata = json.loads((reference_dir / "bank/bank_metadata.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((reference_dir / "bank/bank.pt").read_bytes()).hexdigest() == metadata["bank_sha256"]
    bank = torch.load(reference_dir / "bank/bank.pt", map_location="cpu", weights_only=True)
    assert bank["latents"].shape == (8, *inputs["initial_noise"].shape[1:])
    assert bank["rewards"].shape == (8,) and torch.isfinite(bank["latents"]).all()
    assert torch.isfinite(bank["rewards"]).all()
    assert metadata["config"]["seeds"] == env["bank_seeds"]
    assert metadata["config"]["lookahead_steps"] == 5
    for key in ("model_id", "model_revision", "prompt", "cfg", "target", "reward", "resolution"):
        assert metadata["config"][key] == env[key], f"Banque incompatible : {key}"
    rows = {r["label"]: r for r in json.loads((reference_dir / "results.json").read_text(encoding="utf-8"))}
    assert rows["LiDAR_1"]["beta"] == BETA, "La référence doit être le beta commun, pas le meilleur beta."
    assert rows["LiDAR_1"]["bank_size"] == 8 and rows["LiDAR_1"]["lookahead_steps"] == 5

    folder = reference_dir.parent / datetime.now(timezone.utc).strftime("lidar_ablations_%Y%m%d_%H%M%S_%f_UTC")
    reference = folder / "reference_8x5"
    reference.mkdir(parents=True, exist_ok=False)
    for name in ("environment.json", "algorithm_checks.json", "results.json", "inputs.pt",
                 "BASE_tensors.pt", "BASE.png", "LiDAR_1_tensors.pt", "LiDAR_1.png", "LiDAR_1_steps.json"):
        shutil.copy2(reference_dir / name, reference / name)
    shutil.copytree(reference_dir / "bank", reference / "bank")
    source_dir = folder / "source"
    source_dir.mkdir()
    for path in (Path(__file__), Path(baseline.__file__), Path(rewards.__file__), Path(guidance_lidar.__file__)):
        shutil.copy2(path, source_dir / path.name)
    manifest = dict(reference_dir=str(reference_dir), beta=BETA,
        experiments=[dict(bank_size=16, lookahead_steps=5), dict(bank_size=8, lookahead_steps=20)],
        reference_sha256={str(p.relative_to(reference)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in reference.rglob("*") if p.is_file()},
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_dir.iterdir()},
        current_gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU_TEST_ONLY",
        current_torch=torch.__version__, current_cuda=torch.version.cuda,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        status="prepared", scope="one prompt, one seed, fixed beta; no guarantee of reward improvement")
    assert manifest["deterministic_algorithms"] == env["deterministic_algorithms"]
    save_json(folder / "manifest.json", manifest)
    print("Nouveau dossier :", folder)
    return folder


@torch.no_grad()
def run_bank_ablation(pipe, folder, bank_size, lookahead_steps):
    """Construit une banque, génère une image, puis sauvegarde diagnostics et comparaison."""
    assert (bank_size, lookahead_steps) in ((16, 5), (8, 20))
    folder = Path(folder)
    reference = folder / "reference_8x5"
    env = json.loads((reference / "environment.json").read_text(encoding="utf-8"))
    original_metadata = json.loads((reference / "bank/bank_metadata.json").read_text(encoding="utf-8"))
    original_bank = torch.load(reference / "bank/bank.pt", map_location="cpu", weights_only=True)
    original_rows = {r["label"]: r for r in json.loads((reference / "results.json").read_text(encoding="utf-8"))}
    device = next(pipe.unet.parameters()).device
    inputs = torch.load(reference / "inputs.pt", map_location=device, weights_only=True)
    label = f"bank{bank_size}_steps{lookahead_steps}"
    output = folder / label
    output.mkdir(exist_ok=False)
    bank_dir = output / "bank"
    bank_dir.mkdir()
    seeds = list(range(10000, 10000 + bank_size))
    reuse_count = 8 if lookahead_steps == 5 else 0
    bank_rows = [dict(r, reused=True) for r in original_metadata["rows"]] if reuse_count else []
    latents_list = [original_bank["latents"]] if reuse_count else []
    reward_list = [original_bank["rewards"]] if reuse_count else []
    diagnostics = []
    phase = "bank"
    pipe.scheduler.set_timesteps(lookahead_steps, device=device)
    config = dict(env, bank_size=bank_size, lookahead_steps=lookahead_steps, bank_seeds=seeds,
                  beta=BETA, bank_timesteps=pipe.scheduler.timesteps.cpu().tolist(), reused_candidates=reuse_count)
    save_json(output / "config.json", config)
    save_json(output / "status.json", dict(status="running", phase=phase))

    def decode_and_score(latent):
        image = rewards.decode_latents_differentiable(pipe.vae, latent.float())
        score = rewards.luminance_reward(image, target=env["target"])
        assert score.shape == (1,) and torch.isfinite(score).all() and torch.isfinite(image).all()
        return image.cpu(), score.cpu(), rewards.luminance_map(image).mean().item()

    try:
        if reuse_count:
            for path in (reference / "bank").glob("*.png"):
                shutil.copy2(path, bank_dir / path.name)
        for index in range(reuse_count, bank_size):
            seed = seeds[index]
            noise = baseline.make_initial_noise(pipe, seed, height=env["resolution"][0], width=env["resolution"][1])
            # Même bruit initial par seed ; DDIM repart de zéro avec la nouvelle grille.
            (latent, _), generation_cost = baseline.measure_call(
                baseline.sample_ddim, pipe, noise, inputs["prompt_embeds"], inputs["negative_prompt_embeds"],
                num_steps=lookahead_steps, guidance_scale=env["cfg"])
            (image, score, luminance), scoring_cost = baseline.measure_call(decode_and_score, latent)
            latents_list.append(latent.float().cpu())
            reward_list.append(score.double())
            pixels = (image[0].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")
            Image.fromarray(pixels).save(bank_dir / f"lookahead_{index:02d}_seed{seed}.png")
            bank_rows.append(dict(index=index, seed=seed, reused=False, reward=score.item(), luminance=luminance,
                                  generation_cost=generation_cost, decode_scoring_cost=scoring_cost))
            save_json(bank_dir / "bank_progress.json", bank_rows)
            torch.save(dict(latents=torch.cat(latents_list), rewards=torch.cat(reward_list)), bank_dir / "bank.pt")
            print(f"{label} : candidat {index + 1}/{bank_size}, seed={seed}, L={luminance:.4f}")
            del noise, latent, image, score
        bank_latents_cpu = torch.cat(latents_list)
        bank_rewards_cpu = torch.cat(reward_list)
        assert bank_latents_cpu.shape == (bank_size, *inputs["initial_noise"].shape[1:])
        assert bank_rewards_cpu.shape == (bank_size,) and torch.isfinite(bank_rewards_cpu).all()
        assert torch.isfinite(bank_latents_cpu).all()
        if reuse_count:
            assert torch.equal(bank_latents_cpu[:8], original_bank["latents"])
            assert torch.equal(bank_rewards_cpu[:8], original_bank["rewards"])
        new_rows = [r for r in bank_rows if not r["reused"]]
        generation_seconds = sum(r["generation_cost"]["seconds"] for r in new_rows)
        scoring_seconds = sum(r["decode_scoring_cost"]["seconds"] for r in new_rows)
        incremental_bank_seconds = generation_seconds + scoring_seconds
        reused_bank_seconds = original_metadata["original_build_seconds"] if reuse_count else 0.0
        full_bank_seconds = reused_bank_seconds + incremental_bank_seconds
        bank_metadata = dict(config=config, rows=bank_rows, incremental_generation_seconds=generation_seconds,
            incremental_decode_scoring_seconds=scoring_seconds, incremental_build_seconds=incremental_bank_seconds,
            reused_historical_build_seconds=reused_bank_seconds, full_build_seconds=full_bank_seconds,
            first_eight_preserved_exactly=bool(reuse_count),
            bank_sha256=hashlib.sha256((bank_dir / "bank.pt").read_bytes()).hexdigest())
        save_json(bank_dir / "bank_metadata.json", bank_metadata)

        phase = "guided_generation"
        save_json(output / "status.json", dict(status="running", phase=phase))
        bank_latents = bank_latents_cpu.to(device)
        bank_rewards = bank_rewards_cpu.to(device)

        def generate_and_decode():
            latent, trace, _ = guidance_lidar.sample_lidar(pipe, inputs["initial_noise"], inputs["prompt_embeds"],
                inputs["negative_prompt_embeds"], bank_latents, bank_rewards, beta=BETA,
                num_steps=env["num_steps"], guidance_scale=env["cfg"], return_trace=True, diagnostics=diagnostics)
            image = rewards.decode_latents_differentiable(pipe.vae, latent.float()).cpu()
            return dict(latents=latent.cpu(), trace=trace, image=image)

        data, cost = baseline.measure_call(generate_and_decode)
        torch.save(data, output / "guided_tensors.pt")
        save_json(output / "steps.json", diagnostics)
        with (output / "steps.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
            writer.writeheader()
            writer.writerows(diagnostics)
        assert len(diagnostics) == len(data["trace"]) == env["num_steps"]
        assert all(s["finite"] and s["state_detached"] and s["unet_graph_absent"] for s in diagnostics)
        assert all(torch.isfinite(t).all() for t in data["trace"])
        assert torch.isfinite(data["image"]).all() and torch.isfinite(data["latents"]).all()
        assert all(not p.requires_grad and p.grad is None for m in (pipe.unet, pipe.vae) for p in m.parameters())
        image = data["image"]
        score = rewards.luminance_reward(image, target=env["target"]).item()
        luminance = rewards.luminance_map(image).mean().item()
        peak = max(diagnostics, key=lambda s: s["relative_noise_correction"])
        memory = [c["peak_allocated_mib"] for r in new_rows for c in (r["generation_cost"], r["decode_scoring_cost"])
                  if "peak_allocated_mib" in c]
        if "peak_allocated_mib" in cost:
            memory.append(cost["peak_allocated_mib"])
        result = dict(label=label, beta=BETA, bank_size=bank_size, lookahead_steps=lookahead_steps,
            reward=score, mean_luminance=luminance, absolute_error=abs(luminance - env["target"]),
            delta_reward_vs_base=score - original_rows["BASE"]["reward"],
            delta_reward_vs_reference_8x5=score - original_rows["LiDAR_1"]["reward"],
            ess_reward_below_1_01_steps=sum(s["ess_reward"] < 1.01 for s in diagnostics),
            mean_ess=sum(s["ess"] for s in diagnostics) / len(diagnostics),
            mean_ess_reward=sum(s["ess_reward"] for s in diagnostics) / len(diagnostics),
            unchanged_noise_steps=sum(s["changed_noise_fraction"] == 0 for s in diagnostics),
            max_relative_noise_correction=peak["relative_noise_correction"], peak_timestep=peak["timestep"],
            generation_cost=cost, full_bank_seconds=full_bank_seconds,
            full_bank_plus_one_image_seconds=full_bank_seconds + cost["seconds"],
            incremental_bank_seconds=incremental_bank_seconds,
            incremental_bank_plus_image_seconds=incremental_bank_seconds + cost["seconds"],
            current_run_peak_allocated_mib=max(memory) if memory else None,
            reference_costs_are_historical=True,
            cost_scope="bank generation+decode/scoring and guided sampler+trace+decode; excludes I/O and setup")
        save_json(output / "result.json", result)
        pixels = (image[0].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")
        Image.fromarray(pixels).save(output / "guided.png")
        panel = Image.new("RGB", (3 * 384, 442), "white")
        draw = ImageDraw.Draw(panel)
        entries = [("BASE", reference / "BASE.png", original_rows["BASE"]),
                   ("LiDAR reference 8 x 5", reference / "LiDAR_1.png", original_rows["LiDAR_1"]),
                   (f"LiDAR {bank_size} x {lookahead_steps}", output / "guided.png", result)]
        for col, (title, path, row) in enumerate(entries):
            draw.text((col * 384 + 6, 6), title, fill="black")
            draw.text((col * 384 + 6, 24), f"L={row['mean_luminance']:.6f} R={row['reward']:.6f}", fill="black")
            if col:
                draw.text((col * 384 + 6, 40), f"beta={BETA:.6f}", fill="black")
            with Image.open(path) as preview:
                panel.paste(preview.resize((384, 384)), (col * 384, 58))
        panel.save(output / "comparison.png")
        save_json(output / "status.json", dict(status="passed_for_this_ablation", reward_improvement_required=False))
        print(json.dumps(result, indent=2))
        return result
    except Exception as error:
        save_json(output / "steps.json", diagnostics)
        save_json(output / "failure.json", dict(phase=phase, type=type(error).__name__, message=str(error)))
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        save_json(output / "status.json", dict(status="failed", phase=phase))
        if isinstance(error, torch.cuda.OutOfMemoryError) and device.type == "cuda":
            (output / "OOM.txt").write_text(torch.cuda.memory_summary(device=device), encoding="utf-8")
        raise
    finally:
        # Le notebook conserve son scheduler de génération finale, même après un échec.
        pipe.scheduler.set_timesteps(env["num_steps"], device=device)
