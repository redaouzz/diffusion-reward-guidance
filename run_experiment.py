"""Campagne DDIM reprenable. Les équations restent dans les samplers validés."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np
from PIL import Image
import torch

from baseline import (MODEL_ID, MODEL_REVISION, configure_torch, load_pipeline,
                      encode_prompt_cfg, make_initial_noise, sample_ddim, measure_call)
from guidance_cov import sample_cov
from guidance_full import sample_full
from guidance_lidar import sample_lidar
from rewards import (decode_latents_differentiable, luminance_reward, symmetry_reward,
                     spatial_mask_reward, make_disk_mask, image_reward, load_imagereward)

ROOT = Path(__file__).resolve().parent
METHODS = ("base", "cov", "full", "lidar")
REWARD_DEFAULTS = {"luminance": {"target": 0.7}, "symmetry": {},
                   "spatial_mask": {"radius_fraction": 0.25, "bright": 1.0, "dark": 0.0},
                   "image_reward": {}}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    """Le fichier de statut est publié en dernier, par remplacement atomique."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def completed_result(path):
    """Un JSON complet ET tous ses fichiers intacts ; un PNG seul ne suffit pas."""
    path = Path(path)
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "complete" or not result.get("artifacts"):
            return None
        for name, expected_hash in result["artifacts"].items():
            artifact = (path.parent / name).resolve()
            if not artifact.is_relative_to(path.parent.resolve()) or sha256(artifact) != expected_hash:
                return None
        if result.get("runtime_sha256") or result.get("dependencies"):
            # Un résultat LiDAR dépend aussi des fichiers de banque réellement utilisés.
            if path.name != "result.json" or path.parent.parent.name != "results":
                return None
            campaign = path.parent.parent.parent.resolve()
            if result.get("runtime_sha256") and sha256(campaign / "runtime.json") != result["runtime_sha256"]:
                return None
            for name, expected_hash in result.get("dependencies", {}).items():
                dependency = (campaign / name).resolve()
                if (not dependency.is_relative_to(campaign) or sha256(dependency) != expected_hash
                        or completed_result(dependency) is None):
                    return None
        if result.get("image"):
            with Image.open(path.parent / result["image"]) as image:
                image.verify()
        return result
    except (OSError, ValueError, TypeError, KeyError):
        return None


def read_prompts(path):
    prompts = [line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
               if line.strip()]
    if not prompts or len(prompts) != len(set(prompts)):
        raise ValueError("Le fichier de prompts doit être non vide et sans doublons.")
    return prompts


def load_config(path):
    """Valide la grille avant tout chargement de modèle ou écriture de résultats."""
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    defaults = dict(prompt_limit=None, seeds=[0], methods=list(METHODS),
                    height=512, width=512, num_steps=50, guidance_scale=7.5, tau2=1.0,
                    bank=dict(size=16, num_steps=5, seed_start=10000))
    allowed = set(defaults) | {"prompts_file", "exclude_prompts_file", "split", "rewards"}
    if set(config) - allowed:
        raise ValueError(f"Clés de configuration inconnues : {set(config) - allowed}")
    config = defaults | config
    if config.get("split") not in ("dev", "eval"):
        raise ValueError("split doit être dev ou eval.")
    for key in ("prompts_file", "exclude_prompts_file"):
        if not config.get(key):
            raise ValueError(f"{key} requis pour contrôler la séparation DEV/évaluation.")
        config[key] = str((path.parent / config[key]).resolve())
    prompts, excluded = read_prompts(config["prompts_file"]), read_prompts(config["exclude_prompts_file"])
    if set(prompts) & set(excluded):
        raise ValueError("Chevauchement DEV/évaluation : arrêter avant toute génération.")
    opposite_split = "heldout" if config["split"] == "dev" else "dev"
    protected_file = ROOT / "data" / f"prompts_50_{opposite_split}.txt"
    if protected_file.exists() and set(prompts) & set(read_prompts(protected_file)):
        raise ValueError("Inversion des rôles DEV/held-out : un prompt du split protégé est présent.")
    limit = config["prompt_limit"]
    if limit is not None and (type(limit) is not int or not 1 <= limit <= len(prompts)):
        raise ValueError("prompt_limit doit être positif et ne pas dépasser le fichier.")
    for key, expected in (("height", 512), ("width", 512), ("num_steps", 50),
                          ("guidance_scale", 7.5), ("tau2", 1.0)):
        if config[key] != expected:
            raise ValueError(f"Ce protocole conserve {key}={expected}.")
    seeds = config["seeds"]
    if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds)):
        raise ValueError("Seeds entières, distinctes et comprises entre 0 et 2**63-1 requises.")
    methods = config["methods"]
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError("Méthodes invalides ou dupliquées.")
    if "base" not in methods:
        raise ValueError("BASE est obligatoire pour la comparaison appariée.")
    bank = defaults["bank"] | config["bank"]
    if set(bank) != set(defaults["bank"]) or any(type(x) is not int for x in bank.values()):
        raise ValueError("Configuration de banque invalide.")
    if not 1 <= bank["size"] <= 256 or not 1 <= bank["num_steps"] < 1000:
        raise ValueError("Banque : taille 1..256, pas DDIM 1..999.")
    bank_seeds = set(range(bank["seed_start"], bank["seed_start"] + bank["size"]))
    if min(bank_seeds) < 0 or max(bank_seeds) >= 2**63 or bank_seeds & set(seeds):
        raise ValueError("Seeds de banque invalides ou communes aux générations évaluées.")
    config["bank"] = bank
    rewards = config.get("rewards")
    if not isinstance(rewards, list) or not rewards:
        raise ValueError("Une liste rewards non vide est requise.")
    names = []
    for spec in rewards:
        name = spec.get("name")
        if name not in REWARD_DEFAULTS or set(spec) - {"name", "parameters", "betas"}:
            raise ValueError("Récompense ou clés de récompense inconnues.")
        names.append(name)
        parameters = REWARD_DEFAULTS[name] | spec.get("parameters", {})
        if set(parameters) != set(REWARD_DEFAULTS[name]):
            raise ValueError(f"Paramètres inconnus pour {name}.")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in parameters.values()):
            raise ValueError("Paramètres de récompense finis requis.")
        if name == "luminance" and not 0 <= parameters["target"] <= 1:
            raise ValueError("Cible hors de [0,1].")
        if name == "spatial_mask":
            make_disk_mask(2, 2, **parameters)  # Valide rayon et niveaux avant le GPU.
        betas = spec.get("betas")
        if (not isinstance(betas, list) or not betas or
                any(type(b) not in (int, float) or not math.isfinite(b) or b < 0 for b in betas)
                or len(set(betas)) != len(betas)):
            raise ValueError("Fixer explicitement une grille de beta finis, positifs ou nuls, distincts.")
        spec.update(parameters=parameters, betas=[float(b) for b in betas])
    if len(names) != len(set(names)):
        raise ValueError("Une seule spécification par récompense ; regrouper les beta dans betas.")
    return config


def reward_key(spec):
    return digest_json({"name": spec["name"], "parameters": spec["parameters"]})[:16]


def make_tasks(config):
    """BASE est générée une seule fois par prompt/seed, partagée entre récompenses."""
    prompts = read_prompts(config["prompts_file"])[:config["prompt_limit"]]
    identity = config | {"prompts_file": sha256(config["prompts_file"]),
                         "exclude_prompts_file": sha256(config["exclude_prompts_file"])}
    config_id = digest_json(identity)
    tasks = []
    for index, prompt in enumerate(prompts):
        for seed in config["seeds"]:
            common = dict(config_id=config_id, split=config["split"], prompt_index=index,
                          prompt=prompt, seed=seed)
            tasks.append(common | dict(method="base", reward_name="base", reward_parameters={}, beta=0.0))
            for spec in config["rewards"]:
                for beta in spec["betas"]:
                    for method in METHODS[1:]:
                        if method in config["methods"]:
                            tasks.append(common | dict(method=method, reward_name=spec["name"],
                                                       reward_parameters=spec["parameters"], beta=beta))
    for task in tasks:
        task["result_id"] = digest_json(task)[:24]
    return tasks


def make_reward(spec, prompt, device, reward_model=None):
    """Adaptateur explicite des quatre fonctions existantes, sans nouvelle récompense."""
    name, parameters = spec["name"], spec["parameters"]
    if name == "luminance":
        return lambda images: luminance_reward(images, **parameters)
    if name == "symmetry":
        return symmetry_reward
    if name == "spatial_mask":
        mask = make_disk_mask(512, 512, device=device, **parameters)
        return lambda images: spatial_mask_reward(images, mask)
    if name == "image_reward" and reward_model is not None:
        return lambda images: image_reward(reward_model, prompt, images)
    raise ValueError("Récompense inconnue ou modèle ImageReward absent.")


def place_imagereward(model, device, output):
    """Garder une seule instance ; hors besoin, ses poids attendent en RAM CPU."""
    needs_loading = model is None
    needs_transfer = needs_loading or next(model.parameters()).device != torch.device(device)
    if needs_transfer:
        def prepare():
            reward_model = load_imagereward("cpu") if needs_loading else model
            reward_model.to(device)
            reward_model.device = torch.device(device)
            return reward_model
        model, cost = measure_call(prepare)
        with (output / "setup_costs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(component="ImageReward", loaded=needs_loading,
                                         device=device, cost=cost)) + "\n")
    return model


def commit_files(directory, metadata_name, metadata, tensors=None, image=None, diagnostics=None):
    """Écrit les artefacts puis le manifeste complet avec leurs empreintes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, tensor in (tensors or {}).items():
        temporary = directory / (name + ".tmp")
        torch.save(tensor.detach().cpu(), temporary)
        os.replace(temporary, directory / name)
        artifacts[name] = sha256(directory / name)
    if image is not None:
        # Même convention de quantification que Diffusers numpy_to_pil : round(x*255).
        pixels = (image.detach()[0].float().cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(pixels).save(directory / "image.png.tmp", format="PNG")
        os.replace(directory / "image.png.tmp", directory / "image.png")
        artifacts["image.png"] = sha256(directory / "image.png")
        metadata = metadata | dict(image="image.png")
    if diagnostics is not None:
        atomic_json(directory / "diagnostics.json", diagnostics)
        artifacts["diagnostics.json"] = sha256(directory / "diagnostics.json")
    result = metadata | dict(status="complete", artifacts=artifacts)
    atomic_json(directory / metadata_name, result)
    return result


def total_cost(costs):
    """Les temps s'additionnent, les pics mémoire prennent le maximum."""
    return dict(seconds=sum(cost["seconds"] for cost in costs),
                peak_allocated_mib=max((cost.get("peak_allocated_mib", 0) for cost in costs), default=0) or None)


def prepare_bank(pipe, embeddings, prompt, config, output):
    """Reprise candidat par candidat ; génération indépendante des récompenses/beta."""
    bank_id = digest_json(dict(prompt=prompt, bank=config["bank"], model_revision=MODEL_REVISION))[:24]
    directory = output / "banks" / bank_id
    positive, negative = embeddings
    bank_latents, costs, hashes = [], [], {}
    for index in range(config["bank"]["size"]):
        seed = config["bank"]["seed_start"] + index
        candidate_dir = directory / f"candidate_{index:03d}"
        result = completed_result(candidate_dir / "candidate.json")
        if result is None:
            noise = make_initial_noise(pipe, seed)
            (latent, _), cost = measure_call(sample_ddim, pipe, noise, positive, negative,
                                             num_steps=config["bank"]["num_steps"], guidance_scale=7.5)
            if latent.shape != (1, 4, 64, 64) or not torch.isfinite(latent).all():
                raise FloatingPointError("Latent de banque non fini ou forme invalide.")
            result = commit_files(candidate_dir, "candidate.json", dict(seed=seed, generation_cost=cost),
                                  tensors={"latent.pt": latent.float()})
            del latent, noise
        bank_latents.append(torch.load(candidate_dir / "latent.pt", map_location="cpu", weights_only=True))
        costs.append(result["generation_cost"])
        for name in ("latent.pt", "candidate.json"):
            hashes[f"candidate_{index:03d}/{name}"] = sha256(candidate_dir / name)
    pipe.scheduler.set_timesteps(config["bank"]["num_steps"])
    metadata = dict(status="complete", bank_id=bank_id, prompt=prompt, bank=config["bank"],
                    timesteps=pipe.scheduler.timesteps.tolist(), generation_cost=total_cost(costs), artifacts=hashes)
    atomic_json(directory / "bank.json", metadata)
    return bank_id, torch.cat(bank_latents).to(pipe._execution_device)


def score_bank(pipe, bank_latents, reward_fn, directory, spec):
    """Un décodage/scoring par candidat et récompense, puis partage seeds/beta."""
    key = reward_key(spec)
    score_dir = directory / "scores"
    bank_hash = sha256(directory / "bank.json")
    previous = completed_result(score_dir / (key + ".json"))
    if previous is None or previous.get("bank_sha256") != bank_hash:
        @torch.no_grad()
        def compute():
            values = []
            for latent in bank_latents.split(1):
                value = reward_fn(decode_latents_differentiable(pipe.vae, latent.float()))
                if value.shape != (1,) or not torch.isfinite(value).all():
                    raise FloatingPointError("Récompense de banque non finie ou forme invalide.")
                values.append(value.detach())
            return torch.cat(values)
        values, cost = measure_call(compute)
        commit_files(score_dir, key + ".json", dict(bank_score_key=key, scoring_cost=cost, bank_sha256=bank_hash,
                     reward_name=spec["name"], reward_parameters=spec["parameters"]),
                     tensors={key + ".pt": values})
    return torch.load(score_dir / (key + ".pt"), map_location=pipe._execution_device, weights_only=True)


def runtime_info(pipe, device):
    packages = ("torch", "torchvision", "diffusers", "transformers", "tokenizers",
                "huggingface-hub", "accelerate", "safetensors", "numpy", "Pillow",
                "image-reward", "clip", "timm")
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    sources = {name: sha256(ROOT / name) for name in
               ("baseline.py", "rewards.py", "guidance_cov.py", "guidance_full.py", "guidance_lidar.py", "run_experiment.py")}
    pipe.scheduler.set_timesteps(50)
    return dict(model_id=MODEL_ID, model_revision=MODEL_REVISION, versions=versions,
                python=platform.python_version(), cuda=torch.version.cuda, device=device,
                gpu=torch.cuda.get_device_name() if device == "cuda" else None,
                scheduler={key: value for key, value in pipe.scheduler.config.items() if not key.startswith("_")},
                timesteps=pipe.scheduler.timesteps.tolist(),
                vae_scaling_factor=pipe.vae.config.scaling_factor, vae_dtype=str(next(pipe.vae.parameters()).dtype),
                unet_dtype=str(next(pipe.unet.parameters()).dtype), batch_size=1, eta=0,
                sources=sources, deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                generation_time_scope="DDIM sampler + final FP32 VAE decode; synchronized wall seconds",
                excluded_time="model loading, prompt encoding, PNG/disk IO, final scoring, bank construction")


@contextmanager
def campaign_lock(output):
    """Un seul processus écrivain. Après arrêt brutal, vérifier puis retirer .lock."""
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError("Campagne verrouillée : vérifier qu'aucun job n'écrit, puis retirer .lock si obsolète.")
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    status = "interrupted"
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(f"pid={os.getpid()} host={platform.node()}\n")
        atomic_json(output / "sessions" / f"{stamp}.json", dict(status="running", started_utc=stamp))
        yield
        status = "complete"
    finally:
        lock.unlink(missing_ok=True)
        atomic_json(output / "sessions" / f"{stamp}.json",
                    dict(status=status, started_utc=stamp, wall_seconds=time.perf_counter() - started))


def run_campaign(config, output, device="cuda", dry_run=False):
    tasks = make_tasks(config)
    summary = dict(config_id=tasks[0]["config_id"], task_count=len(tasks),
                   prompt_count=len({task["prompt_index"] for task in tasks}),
                   methods=config["methods"], rewards=config["rewards"],
                   images_per_prompt=len(tasks) // len({task["prompt_index"] for task in tasks}))
    if dry_run:
        print(json.dumps(summary, indent=2))
        return summary
    output = Path(output).resolve()
    with campaign_lock(output):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous["config_id"] != summary["config_id"]:
                raise ValueError("Configuration différente : utiliser un nouveau répertoire de campagne.")
        else:
            atomic_json(manifest_path, summary | dict(config=config, created_utc=datetime.now(timezone.utc).isoformat()))
        pending = []
        for task in tasks:
            result = completed_result(output / "results" / task["result_id"] / "result.json")
            if result is None or any(result.get(key) != value for key, value in task.items()):
                pending.append(task)
        if not pending:
            print(f"{len(tasks)}/{len(tasks)} résultats intacts ; aucun modèle chargé.")
            return summary | dict(completed=len(tasks), generated=0)
        configure_torch()
        pipe, loading_cost = measure_call(load_pipeline, device)
        with (output / "setup_costs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(component="SD1.5", loaded=True, device=device, cost=loading_cost)) + "\n")
        pipe.vae.to(dtype=torch.float32).eval().requires_grad_(False)
        runtime = runtime_info(pipe, device)
        runtime_path = output / "runtime.json"
        if runtime_path.exists():
            if json.loads(runtime_path.read_text(encoding="utf-8")) != runtime:
                raise ValueError("Environnement/sources différents : créer une campagne séparée pour éviter de mélanger les résultats.")
        else:
            atomic_json(runtime_path, runtime)
        runtime_hash = sha256(runtime_path)
        reward_model = None
        current_prompt, embeddings, bank_id, bank_latents = None, None, None, None
        generated = 0
        for task in pending:
            diagnostics = []
            try:
                if task["prompt"] != current_prompt:
                    current_prompt = task["prompt"]
                    embeddings = encode_prompt_cfg(pipe, current_prompt)
                    bank_id, bank_latents = None, None
                noise = make_initial_noise(pipe, task["seed"])
                spec = dict(name=task["reward_name"], parameters=task["reward_parameters"])
                reward_fn, bank_rewards, score_key = None, None, None
                if task["method"] != "base":
                    if spec["name"] == "image_reward":
                        reward_model = place_imagereward(reward_model, device, output)
                    # Libérer sa VRAM sans relire le checkpoint à chaque seed/récompense.
                    if spec["name"] != "image_reward" and reward_model is not None:
                        reward_model = place_imagereward(reward_model, "cpu", output)
                    reward_fn = make_reward(spec, current_prompt, device, reward_model)
                elif reward_model is not None:
                    reward_model = place_imagereward(reward_model, "cpu", output)
                if task["method"] == "lidar":
                    if bank_latents is None:
                        bank_id, bank_latents = prepare_bank(pipe, embeddings, current_prompt, config, output)
                    bank_rewards = score_bank(pipe, bank_latents, reward_fn, output / "banks" / bank_id, spec)
                    score_key = reward_key(spec)

                def generate():
                    positive, negative = embeddings
                    arguments = dict(num_steps=50, guidance_scale=7.5)
                    if task["method"] == "base":
                        latents, _ = sample_ddim(pipe, noise, positive, negative, **arguments)
                    elif task["method"] == "cov":
                        latents, _, _ = sample_cov(pipe, noise, positive, negative, reward_fn, task["beta"],
                                                   tau2=1.0, diagnostics=diagnostics, **arguments)
                    elif task["method"] == "full":
                        latents, _, _ = sample_full(pipe, noise, positive, negative, reward_fn, task["beta"],
                                                    diagnostics=diagnostics, **arguments)
                    else:
                        latents, _, _ = sample_lidar(pipe, noise, positive, negative, bank_latents, bank_rewards,
                                                     task["beta"], diagnostics=diagnostics, **arguments)
                    if latents.shape != (1, 4, 64, 64) or not torch.isfinite(latents).all():
                        raise FloatingPointError("Latents finaux non finis ou forme invalide.")
                    with torch.no_grad():
                        images = decode_latents_differentiable(pipe.vae, latents.float())
                    if images.shape != (1, 3, 512, 512) or not torch.isfinite(images).all():
                        raise FloatingPointError("Image finale non finie ou forme invalide.")
                    return latents, images

                (latents, images), cost = measure_call(generate)
                value, scoring_cost, baseline_rewards = None, None, {}
                if reward_fn is not None:
                    with torch.no_grad():
                        reward, scoring_cost = measure_call(reward_fn, images)
                    if reward.shape != (1,) or not torch.isfinite(reward).all():
                        raise FloatingPointError("Récompense finale non finie ou forme invalide.")
                    value = reward.item()
                    del reward
                else:
                    # BASE sert aux quatre récompenses, sans générer quatre fois la même image.
                    for baseline_spec in config["rewards"]:
                        if baseline_spec["name"] == "image_reward":
                            reward_model = place_imagereward(reward_model, device, output)
                        baseline_fn = make_reward(baseline_spec, current_prompt, device, reward_model)
                        with torch.no_grad():
                            baseline_value, baseline_cost = measure_call(baseline_fn, images)
                        if baseline_value.shape != (1,) or not torch.isfinite(baseline_value).all():
                            raise FloatingPointError("Récompense BASE non finie ou forme invalide.")
                        baseline_rewards[reward_key(baseline_spec)] = dict(
                            reward_name=baseline_spec["name"], reward_parameters=baseline_spec["parameters"],
                            reward_tensor=baseline_value.item(), scoring_cost=baseline_cost)
                        del baseline_value, baseline_fn
                dependencies = {}
                if task["method"] == "lidar":
                    for name in (f"banks/{bank_id}/bank.json", f"banks/{bank_id}/scores/{score_key}.json"):
                        dependencies[name] = sha256(output / name)
                metadata = task | dict(reward_tensor=value, generation_cost=cost, scoring_cost=scoring_cost,
                                       baseline_rewards=baseline_rewards,
                                       dependencies=dependencies,
                                       bank_id=bank_id if task["method"] == "lidar" else None,
                                       bank_score_key=score_key, runtime_sha256=runtime_hash,
                                       completed_utc=datetime.now(timezone.utc).isoformat())
                commit_files(output / "results" / task["result_id"], "result.json", metadata,
                             tensors={"latents.pt": latents}, image=images, diagnostics=diagnostics)
                generated += 1
                print(f"[{len(tasks)-len(pending)+generated}/{len(tasks)}] prompt={task['prompt_index']} seed={task['seed']} "
                      f"{task['method']} {task['reward_name']} beta={task['beta']:g} "
                      f"R={value if value is not None else {v['reward_name']: v['reward_tensor'] for v in baseline_rewards.values()}}", flush=True)
                del latents, images, noise, bank_rewards, reward_fn
            except Exception as error:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                atomic_json(output / "failures" / f"{stamp}_{task['result_id']}.json",
                            dict(task=task, error_type=type(error).__name__, message=str(error),
                                 traceback=traceback.format_exc(), diagnostics=diagnostics))
                raise  # Pas de suppression des échecs ni de changement automatique de beta.
        return summary | dict(completed=len(tasks), generated=generated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Valider/compter sans poids ni génération.")
    args = parser.parse_args()
    run_campaign(load_config(args.config), args.output, args.device, args.dry_run)


if __name__ == "__main__":
    main()
