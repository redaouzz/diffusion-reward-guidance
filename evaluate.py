"""Évaluer les PNG validés, sans charger SD1.5 ni générer de nouvelles images."""

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image
import torch

from rewards import (image_reward, load_imagereward, luminance_reward,
                     make_disk_mask, spatial_mask_reward, symmetry_reward,
                     IMAGEREWARD_CODE_REVISION, IMAGEREWARD_MODEL_REVISION,
                     IMAGEREWARD_WEIGHTS_SHA256, BERT_REVISION)

CLIP_REVISION = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
CLIP_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
HPS_CODE_REVISION = "866735ecaae999fa714bd9edfa05aa2672669ee3"
HPS_MODEL_REVISION = "697403c78157020a1ae59d23f111aa58ced35b0a"
HPS_SHA256 = "07cb0d20626232204ca3a6ec53ff5984e7e53212a8a7b0e0ec2bac1bdaf125a1"
FIELDS = ["result_id", "config_id", "split", "prompt_index", "prompt", "seed", "method",
          "reward_name", "reward_parameters_json", "beta", "image_path", "reward_tensor",
          "reward_png", "clip_score", "hpsv2", "seconds", "peak_allocated_mib",
          "bank_id", "bank_score_key", "bank_generation_seconds", "bank_scoring_seconds"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def require_source_revision(distribution, revision):
    """Refuser une version non identifiée au lieu de prétendre utiliser le pin annoncé."""
    raw = importlib.metadata.distribution(distribution).read_text("direct_url.json")
    if raw is None or revision not in raw:
        raise RuntimeError(f"Réinstaller {distribution} au commit {revision} épinglé dans le Dockerfile.")


def metric_spec(metric, device):
    common = {"device": str(device), "input": "RGB PNG 8-bit", "batch_size": 1,
              "gpu": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
              "cuda": torch.version.cuda,
              "evaluate_sha256": sha256_file(__file__)}
    if metric == "clip":
        return {**common, "model": "OpenAI ViT-B/32", "code_revision": CLIP_REVISION,
                "weights_sha256": CLIP_SHA256, "prefix": "A photo depicts ",
                "preprocess": "official PIL bicubic Resize(224), CenterCrop(224), CLIP normalize",
                "precision": "fp16 CUDA / fp32 CPU; feature normalization fp32",
                "definition": "2.5 * max(cosine(unit image feature, unit text feature), 0)",
                "tokenizer": "CLIP BPE; context 77; truncate=True"}
    if metric == "hps":
        return {**common, "model": "HPSv2 v2.0 ViT-H-14", "code_revision": HPS_CODE_REVISION,
                "model_revision": HPS_MODEL_REVISION, "weights_sha256": HPS_SHA256,
                "preprocess": "official light_augmentation=True: RGB tensor, bicubic longest edge 224, zero pad, CLIP normalize",
                "precision": "fp32 parameters; CUDA autocast fp16 / CPU fp32",
                "definition": "normalized feature dot product; raw score, no x100",
                "tokenizer": "HPS bundled CLIP BPE; context 77; no prefix"}
    return {**common, "rewards_sha256": sha256_file(Path(__file__).with_name("rewards.py")),
            "definition": "same rewards.py functions as guidance, evaluated on quantized PNG",
            "imagereward_code_revision": IMAGEREWARD_CODE_REVISION,
            "imagereward_model_revision": IMAGEREWARD_MODEL_REVISION,
            "imagereward_weights_sha256": IMAGEREWARD_WEIGHTS_SHA256,
            "bert_revision": BERT_REVISION,
            "imagereward_preprocess": "tensor bicubic antialias Resize(224), CenterCrop, CLIP normalize; score_gard; fp32"}


def load_clip_scorer(device, cache_dir):
    require_source_revision("clip", CLIP_REVISION)
    import clip

    model, preprocess = clip.load("ViT-B/32", device=device, jit=False,
                                  download_root=str(Path(cache_dir) / "clip") if cache_dir else None)
    model.eval().requires_grad_(False)

    def score(path, prompt):
        with Image.open(path) as image:
            pixels = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
        tokens = clip.tokenize(["A photo depicts " + prompt], truncate=True).to(device)
        with torch.no_grad():
            image_features = model.encode_image(pixels).float()
            text_features = model.encode_text(tokens).float()
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
            return (2.5 * (image_features * text_features).sum(-1).clamp_min(0)).item()

    return score


def load_hps_scorer(device, cache_dir):
    require_source_revision("hpsv2", HPS_CODE_REVISION)
    from huggingface_hub import hf_hub_download
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

    checkpoint_path = hf_hub_download("xswu/HPSv2", "HPS_v2_compressed.pt",
                                       revision=HPS_MODEL_REVISION, cache_dir=cache_dir)
    if sha256_file(checkpoint_path) != HPS_SHA256:
        raise RuntimeError("Le checkpoint HPSv2 ne correspond pas au SHA-256 figé.")
    # Le checkpoint HPS contient tous les poids : inutile de télécharger d'abord LAION.
    # strict=True interdit de scorer avec des paramètres restés à leur valeur aléatoire.
    model, _, preprocess = create_model_and_transforms(
        "ViT-H-14", pretrained=None, precision="fp32", device="cpu", jit=False,
        light_augmentation=True, output_dict=True,
        image_mean=(0.48145466, 0.4578275, 0.40821073),
        image_std=(0.26862954, 0.26130258, 0.27577711),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    model.to(device).eval().requires_grad_(False)
    tokenizer = get_tokenizer("ViT-H-14")

    def score(path, prompt):
        with Image.open(path) as image:
            pixels = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
        tokens = tokenizer([prompt]).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                             enabled=torch.device(device).type == "cuda"):
            features = model(pixels, tokens)
            # L'architecture officielle normalise déjà les deux vecteurs.
            similarity = features["image_features"] @ features["text_features"].T
        return similarity[0, 0].item()

    return score


def reward_from_png(path, prompt, name, parameters, model=None, device="cpu"):
    with Image.open(path) as image:
        pixels = torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1)
    pixels = pixels.unsqueeze(0).to(device=device, dtype=torch.float32) / 255
    with torch.no_grad():
        if name == "luminance":
            value = luminance_reward(pixels, **parameters)
        elif name == "symmetry":
            if parameters:
                raise ValueError("La symétrie n'a pas de paramètre.")
            value = symmetry_reward(pixels)
        elif name == "spatial_mask":
            mask = make_disk_mask(*pixels.shape[-2:], device=device, **parameters)
            value = spatial_mask_reward(pixels, mask)
        elif name == "image_reward":
            if parameters or model is None:
                raise ValueError("ImageReward requiert son modèle figé, sans autre paramètre.")
            value = image_reward(model, prompt, pixels)
        else:
            raise ValueError(f"Récompense inconnue : {name}")
    return value.item()


def collect_rows(root):
    """Lire seulement les résultats complets validés; BASE est évalué pour chaque récompense."""
    from run_experiment import completed_result

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    reward_specs = manifest["config"]["rewards"]
    rows = []
    banks = {}
    for result_path in sorted((root / "results").glob("*/result.json")):
        result = completed_result(result_path)
        if result is None:
            print(f"Ignoré (incomplet/corrompu) : {result_path.parent.name}")
            continue
        specs = reward_specs if result["method"].lower() == "base" else [
            {"name": result["reward_name"], "parameters": result["reward_parameters"]}]
        cost = result.get("generation_cost", {})
        bank_id = result.get("bank_id")
        bank_cost = {}
        score_cost = {}
        if bank_id:
            bank_path = root / "banks" / bank_id / "bank.json"
            if bank_id not in banks:
                banks[bank_id] = completed_result(bank_path)
            bank = banks[bank_id]
            if bank is None:
                raise ValueError(f"Banque absente ou corrompue : {bank_id}")
            bank_cost = bank.get("generation_cost", {})
            # Le fichier de score porte sa propre clé et sa propre durée.
            score_path = root / "banks" / bank_id / "scores" / f"{result['bank_score_key']}.json"
            score = completed_result(score_path)
            if score is None or score.get("bank_sha256") != sha256_file(bank_path):
                raise ValueError(f"Scoring absent ou incompatible avec la banque : {score_path}")
            score_cost = score.get("scoring_cost", {})
        for spec in specs:
            row = {key: result.get(key) for key in ("config_id", "split", "prompt_index", "prompt",
                                                   "seed", "method", "beta", "reward_tensor")}
            if result["method"].lower() == "base":
                row["reward_tensor"] = None
                for baseline_score in result.get("baseline_rewards", {}).values():
                    if (baseline_score["reward_name"] == spec["name"]
                            and baseline_score["reward_parameters"] == spec.get("parameters", {})):
                        row["reward_tensor"] = baseline_score["reward_tensor"]
            row.update(result_id=result_path.parent.name, reward_name=spec["name"],
                       reward_parameters_json=json.dumps(spec.get("parameters", {}), sort_keys=True),
                       image_path=str(result_path.parent / result.get("image", "image.png")),
                       reward_png=None, clip_score=None, hpsv2=None,
                       seconds=cost.get("seconds"), peak_allocated_mib=cost.get("peak_allocated_mib"),
                       bank_id=bank_id, bank_score_key=result.get("bank_score_key"),
                       bank_generation_seconds=bank_cost.get("seconds"),
                       bank_scoring_seconds=score_cost.get("seconds"))
            rows.append(row)
    return rows


def write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def score_pass(rows, metric, scorer, root, specification):
    """Cache par image+prompt+définition; écriture après chaque score, aucune image en RAM globale."""
    column = {"reward": "reward_png", "clip": "clip_score", "hps": "hpsv2"}[metric]
    cache = Path(root) / "evaluation_cache" / metric
    cache.mkdir(parents=True, exist_ok=True)
    for row in rows:
        identity = {"specification": specification, "image_sha256": sha256_file(row["image_path"]),
                    "prompt": row["prompt"]}
        if metric == "reward":
            identity.update(reward_name=row["reward_name"], parameters=row["reward_parameters_json"])
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = cache / f"{key}.json"
        value = None
        if path.exists():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved.get("identity") == identity and math.isfinite(float(saved["value"])):
                    value = float(saved["value"])
            except (ValueError, TypeError, KeyError):
                pass
        if value is None:
            if scorer is None:
                continue  # Relecture seule des métriques calculées lors d'un appel antérieur.
            started = time.perf_counter()
            value = float(scorer(row))
            if not math.isfinite(value):
                raise ValueError(f"Score {metric} non fini : {row['result_id']}")
            atomic_json(path, {"identity": identity, "value": value,
                               "seconds": time.perf_counter() - started})
        row[column] = value


def evaluate(root, metrics=("reward", "clip", "hps"), device="cuda", cache_dir=None):
    root = Path(root)
    rows = collect_rows(root)
    if not rows:
        raise ValueError("Aucun résultat complet validé à évaluer.")
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponible : choisir explicitement --device cpu pour un contrôle CPU.")
    from baseline import configure_torch
    configure_torch()
    versions = {}
    for name in ("torch", "torchvision", "Pillow", "numpy", "transformers", "clip", "hpsv2"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    provenance = {"python": platform.python_version(), "versions": versions, "metrics": {},
                  "device": device, "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
                  "cuda": torch.version.cuda, "rows": len(rows), "complete_metrics": []}
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    planned = manifest.get("task_count")
    completed = len({row["result_id"] for row in rows})
    if planned is not None and completed > planned:
        raise ValueError("Le nombre de générations complètes dépasse la matrice déclarée.")
    provenance.update(planned_generations=planned, completed_generations=completed,
                      missing_generations=planned - completed if planned is not None else None,
                      generation_complete=completed == planned if planned is not None else None,
                      failure_records_count=len(list((root / "failures").glob("*.json"))),
                      coverage_note="Scores describe valid completed generations only. Failure records are historical attempts, not a final failure rate; an interrupted attempt may later succeed.")
    provenance_path = root / "evaluation_provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        for metric in previous.get("complete_metrics", []):
            if metric in metrics:
                continue
            specification = {**metric_spec(metric, device), "versions": versions}
            if previous["metrics"].get(metric) == specification:
                score_pass(rows, metric, None, root, specification)
                column = {"reward": "reward_png", "clip": "clip_score", "hps": "hpsv2"}[metric]
                if all(row[column] is not None for row in rows):
                    provenance["metrics"][metric] = specification
                    provenance["complete_metrics"].append(metric)
    for metric in metrics:
        specification = {**metric_spec(metric, device), "versions": versions}
        provenance["metrics"][metric] = specification
        atomic_json(root / "evaluation_provenance.json", provenance)
        if metric == "reward":
            simple = [row for row in rows if row["reward_name"] != "image_reward"]
            score_pass(simple, metric, lambda row: reward_from_png(
                row["image_path"], row["prompt"], row["reward_name"],
                json.loads(row["reward_parameters_json"])), root, specification)
            image_rows = [row for row in rows if row["reward_name"] == "image_reward"]
            if image_rows:
                require_source_revision("image-reward", IMAGEREWARD_CODE_REVISION)
                model = load_imagereward(device=device, cache_dir=cache_dir)
                score_pass(image_rows, metric, lambda row: reward_from_png(
                    row["image_path"], row["prompt"], row["reward_name"], {}, model, device), root, specification)
                del model
        else:
            loader = load_clip_scorer if metric == "clip" else load_hps_scorer
            score = loader(device, cache_dir)
            score_pass(rows, metric, lambda row: score(row["image_path"], row["prompt"]), root, specification)
            del score
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        provenance["complete_metrics"].append(metric)
        write_csv(root / "evaluation.csv", rows)
        atomic_json(root / "evaluation_provenance.json", provenance)
        print(f"{metric} : {len(rows)} lignes évaluées/cache validé.")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Répertoire de campagne contenant manifest.json.")
    parser.add_argument("--metrics", nargs="+", choices=("reward", "clip", "hps"), default=["reward", "clip", "hps"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cache-dir", type=Path, help="Cache des poids, idéalement sur volume persistant.")
    args = parser.parse_args()
    evaluate(args.input, list(dict.fromkeys(args.metrics)), args.device, args.cache_dir)


if __name__ == "__main__":
    main()
