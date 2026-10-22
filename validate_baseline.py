"""Validation commune au notebook et à Northflank, sans guidance de récompense."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from baseline import (
    MODEL_ID, MODEL_REVISION, configure_torch, load_pipeline,
    encode_prompt_cfg, make_initial_noise, sample_ddim,
    sample_official, decode_latents, measure_call,
)


# Égalité exacte attendue à arithmétique identique ; marge stricte et fixée avant
# exécution. 1e-5 est bien inférieur à un ulp fp16 autour de 1 (~9.77e-4).
# Elle ne justifie pas une divergence accumulée. Ne pas l'élargir pour réussir.
ATOL = 1e-5
RTOL = 1e-5


def compare_tensors(actual, expected):
    """Diagnostics en fp32, avant quantification des images en PNG."""
    actual = actual.detach().cpu().float()
    expected = expected.detach().cpu().float()
    if actual.shape != expected.shape:
        raise AssertionError(f"Formes différentes : {actual.shape} et {expected.shape}.")
    error = actual - expected
    return {
        "shape": list(actual.shape),
        "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
        "exact_equal": torch.equal(actual, expected),
        "max_absolute_error": error.abs().max().item(),
        "rmse": error.square().mean().sqrt().item(),
        "relative_l2_error": (error.norm() / expected.norm().clamp_min(1e-12)).item(),
        "within_tolerance": torch.allclose(actual, expected, atol=ATOL, rtol=RTOL),
    }


def select_dev_prompts(count=5):
    """Préfixe du jeu dev fourni ; vérifie la provenance et exclut le heldout."""
    data_dir = Path(__file__).resolve().parent / "data"
    all_prompts = (data_dir / "generation_prompts.txt").read_text(encoding="utf-8-sig").splitlines()
    dev = (data_dir / "prompts_50_dev.txt").read_text(encoding="utf-8-sig").splitlines()
    heldout = (data_dir / "prompts_50_heldout.txt").read_text(encoding="utf-8-sig").splitlines()
    all_prompts = [p.strip() for p in all_prompts if p.strip()]
    dev = [p.strip() for p in dev if p.strip()]
    heldout = [p.strip() for p in heldout if p.strip()]
    if set(dev) & set(heldout):
        raise ValueError("Les listes dev et heldout se recouvrent.")
    if not set(dev).issubset(set(all_prompts)) or not 1 <= count <= len(dev):
        raise ValueError("Sélection de prompts dev invalide.")
    return dev[:count]


def save_environment(pipe, output_dir):
    """Enregistre versions réelles, révision des poids, scheduler et sources."""
    output_dir = Path(output_dir)
    if (output_dir / "environment.json").exists():
        raise FileExistsError("Ce run existe déjà. Choisir un nouveau dossier pour conserver ses résultats.")
    output_dir.mkdir(parents=True, exist_ok=True)
    pipe.scheduler.set_timesteps(50, device=pipe._execution_device)
    config = pipe.scheduler.config
    assert type(pipe.scheduler).__name__ == "DDIMScheduler"
    assert config.prediction_type == "epsilon" and config.timestep_spacing == "leading"
    assert config.steps_offset == 1 and config.set_alpha_to_one is False
    assert config.clip_sample is False and config.thresholding is False
    assert pipe.scheduler.timesteps.tolist() == list(range(981, 0, -20))
    assert torch.equal(pipe.scheduler.final_alpha_cumprod, pipe.scheduler.alphas_cumprod[0])
    project_dir = Path(__file__).resolve().parent
    versions = {}
    for package in ("torch", "diffusers", "transformers", "accelerate", "huggingface-hub",
                    "safetensors", "numpy", "Pillow"):
        versions[package] = importlib.metadata.version(package)
    source_hashes = {}
    for name in ("baseline.py", "validate_baseline.py", "requirements.txt",
                 "data/prompts_50_dev.txt", "data/prompts_50_heldout.txt", "data/generation_prompts.txt"):
        source_hashes[name] = hashlib.sha256((project_dir / name).read_bytes()).hexdigest()
    environment = {
        "python": sys.version, "platform": platform.platform(), "versions": versions,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "weight_variant": "fp16",
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_mib": torch.cuda.get_device_properties(0).total_memory / 2**20 if torch.cuda.is_available() else None,
        "dtype": str(pipe.unet.dtype), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "scheduler_class": type(pipe.scheduler).__name__,
        "scheduler": dict(pipe.scheduler.config), "timesteps": pipe.scheduler.timesteps.tolist(),
        "vae_scaling_factor": pipe.vae.config.scaling_factor,
        "vae_scale_factor": pipe.vae_scale_factor, "atol": ATOL, "rtol": RTOL,
        "height": 512, "width": 512, "steps": 50, "eta": 0.0, "cfg": 7.5,
        "source_sha256": source_hashes,
        "timing_scope": "generation + decode, including CPU trace copies; excludes text encoding and model loading",
    }
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (output_dir / "requirements-observed.txt").write_text(freeze, encoding="utf-8")
    if torch.cuda.is_available():
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
        (output_dir / "nvidia-smi.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    return environment


def validate_prompt(pipe, prompt, seed, output_dir, repeat=False):
    """Compare embeddings, trajectoire DDIM, image flottante et répétabilité."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    positive, negative = encode_prompt_cfg(pipe, prompt)
    with torch.no_grad():
        official_positive, official_negative = pipe.encode_prompt(
            prompt, pipe._execution_device, 1, True, negative_prompt="",
        )
    positive_check = compare_tensors(positive, official_positive)
    negative_check = compare_tensors(negative, official_negative)
    initial_noise = make_initial_noise(pipe, seed)
    assert initial_noise.shape == (1, 4, 64, 64)
    assert positive.shape == negative.shape == (1, 77, 768)
    assert torch.isfinite(initial_noise).all()
    assert torch.equal(initial_noise, make_initial_noise(pipe, seed))

    official, official_cost = measure_call(
        sample_official, pipe, initial_noise, positive, negative, return_trace=True,
    )
    official_latents, official_image, official_trace = official
    official_image = official_image.detach().cpu()
    del official
    timesteps = pipe.scheduler.timesteps.detach().cpu().clone()
    scheduler_config = dict(pipe.scheduler.config)

    def our_generation():
        latents, trace = sample_ddim(pipe, initial_noise, positive, negative, return_trace=True)
        image = decode_latents(pipe, latents)
        return latents, image, trace

    custom, custom_cost = measure_call(our_generation)
    custom_latents, custom_image, custom_trace = custom
    custom_latents = custom_latents.detach().cpu()
    custom_image = custom_image.detach().cpu()
    del custom
    assert torch.equal(timesteps, pipe.scheduler.timesteps.cpu())
    assert scheduler_config == dict(pipe.scheduler.config)
    assert len(official_trace) == len(custom_trace) == 50

    trajectory = []
    for index, (actual, expected) in enumerate(zip(custom_trace, official_trace)):
        trajectory.append({"step": index, "timestep": int(timesteps[index]), **compare_tensors(actual, expected)})
    latent_check = compare_tensors(custom_latents, official_latents)
    image_check = compare_tensors(custom_image, official_image)
    repetition = {"executed": False}
    if repeat:
        # Refaire chaque chemin distingue nondéterminisme et erreur de la boucle.
        repeated_reference = sample_official(pipe, initial_noise, positive, negative, return_trace=True)
        repetition = {
            "executed": True,
            "official": compare_tensors(repeated_reference[0], official_latents),
            "official_image": compare_tensors(repeated_reference[1], official_image),
        }
        del repeated_reference
        repeated_latents, _ = sample_ddim(pipe, initial_noise, positive, negative)
        repetition["custom"] = compare_tensors(repeated_latents, custom_latents)
        del repeated_latents

    checks = [positive_check, negative_check, latent_check, image_check, *trajectory]
    passed = all(c["finite"] and c["within_tolerance"] for c in checks)
    if repeat:
        passed = passed and all(repetition[key]["exact_equal"] for key in ("official", "official_image", "custom"))
    report = {
        "prompt": prompt, "seed": seed, "passed": passed, "atol": ATOL, "rtol": RTOL,
        "embeddings_positive": positive_check, "embeddings_negative": negative_check,
        "final_latents": latent_check, "images_before_png": image_check,
        "first_failed_step": next((r["step"] for r in trajectory if not r["finite"] or not r["within_tolerance"]), None),
        "trajectory": trajectory, "repeatability": repetition,
        "official_cost": official_cost, "custom_cost": custom_cost,
    }
    (output_dir / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save({
        "initial_noise": initial_noise.cpu(), "prompt_embeds": positive.cpu(),
        "negative_prompt_embeds": negative.cpu(), "official_latents": official_latents,
        "custom_latents": custom_latents,
    }, output_dir / "tensors.pt")
    for name, tensor in (("official", official_image), ("custom", custom_image)):
        pixels = (tensor[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(pixels).save(output_dir / f"{name}.png")
    if not passed:
        raise AssertionError(
            f"Validation échouée : premier pas divergent={report['first_failed_step']}. "
            f"Consulter {output_dir / 'validation.json'} ; ne pas élargir automatiquement la tolérance."
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/baseline"))
    args = parser.parse_args()
    configure_torch()
    prompts = select_dev_prompts(args.count)
    pipe = load_pipeline()
    save_environment(pipe, args.output)
    # Chauffe U-Net + VAE, exclue des chronométrages de validation.
    positive, negative = encode_prompt_cfg(pipe, prompts[0])
    initial_noise = make_initial_noise(pipe, args.seed)
    warm_latents, _ = sample_ddim(pipe, initial_noise, positive, negative, num_steps=1)
    decode_latents(pipe, warm_latents)
    del warm_latents
    reports = []
    for index, prompt in enumerate(prompts):
        print(f"[{index + 1}/{len(prompts)}] {prompt}", flush=True)
        report = validate_prompt(pipe, prompt, args.seed, args.output / f"prompt_{index:02d}", repeat=index == 0)
        reports.append(report)
        print(f"PASS | erreur latente max={report['final_latents']['max_absolute_error']:.3g}", flush=True)
    (args.output / "summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
