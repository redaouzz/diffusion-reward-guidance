"""Récompenses et chemin VAE différentiable séparé de BASE."""

import torch

IMAGEREWARD_CODE_REVISION = "78f8051059b34fef8e0032ccb40561c23a2ddbbb"
IMAGEREWARD_MODEL_REVISION = "5736be03b2652728fb87788c9797b0570450ab72"
IMAGEREWARD_WEIGHTS_SHA256 = "d71dc8d4912bc633c06ab8ef1ef6a81e8199d4369f14b47b29baf1d19de561f6"
BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
IMAGEREWARD_FD_STEPS = (1e-3, 3e-4, 1e-4, 3e-5, 1e-5)


def _float_images(images):
    if images.ndim != 4 or images.shape[1] != 3 or min(images.shape) < 1:
        raise ValueError("Images attendues : [B, 3, H, W].")
    if not images.is_floating_point() or not torch.isfinite(images).all():
        raise ValueError("Les images doivent être des tenseurs flottants finis.")
    # Les petites erreurs quadratiques sont réduites au moins en fp32.
    return images.float() if images.dtype in (torch.float16, torch.bfloat16) else images


def luminance_map(images):
    """Carte [B,H,W] : coefficients RGB fixes, sans correction gamma."""
    images = _float_images(images)
    return 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]


def luminance_reward(images, target=0.7):
    """R_lum = -(moyenne de luminance - cible)^2, résultat [B]."""
    if not 0 <= target <= 1:
        raise ValueError("La cible de luminance doit appartenir à [0,1].")
    mean_luminance = luminance_map(images).mean(dim=(1, 2))
    return -(mean_luminance - target).square()


def symmetry_reward(images):
    """R_sym = -||y-flip(y)||²/(3HW), retournement de la largeur seulement."""
    images = _float_images(images)
    return -(images - images.flip(-1)).square().mean(dim=(1, 2, 3))


def spatial_mask_reward(images, mask):
    """R_mask = -||luminance(y)-masque||²/(HW), résultat [B]."""
    luminance = luminance_map(images)
    if mask.ndim != 2 or tuple(mask.shape) != tuple(luminance.shape[-2:]):
        raise ValueError("Le masque doit avoir la forme [H,W] de l'image.")
    if not torch.isfinite(mask).all() or not ((mask >= 0) & (mask <= 1)).all():
        raise ValueError("Le masque doit être fini et dans [0,1].")
    mask = mask.to(device=luminance.device, dtype=luminance.dtype)
    return -(luminance - mask).square().mean(dim=(1, 2))


def make_disk_mask(height, width, radius_fraction=0.25, bright=1.0, dark=0.0,
                   device="cpu", dtype=torch.float32):
    """Disque centré, rayon = radius_fraction * min(H,W), centres des pixels."""
    if height < 1 or width < 1 or not 0 < radius_fraction <= 0.5:
        raise ValueError("Dimensions positives et fraction de rayon dans ]0,0.5] requises.")
    if not 0 <= dark <= bright <= 1:
        raise ValueError("Niveaux attendus : 0 <= dark <= bright <= 1.")
    rows = torch.arange(height, device=device, dtype=dtype) - (height - 1) / 2
    columns = torch.arange(width, device=device, dtype=dtype) - (width - 1) / 2
    inside = rows[:, None].square() + columns[None, :].square() <= (radius_fraction * min(height, width)) ** 2
    return torch.where(inside, bright, dark).to(dtype=dtype)


def decode_latents_differentiable(vae, latents):
    """Même décodage que BASE ; garde le graphe vers les latents, poids VAE gelés."""
    if any(parameter.requires_grad for parameter in vae.parameters()):
        raise ValueError("Geler le VAE avec vae.eval().requires_grad_(False) avant le décodage.")
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    # Garder cet ordre, notamment le cast après clamp, pour la parité fp16 avec BASE.
    return (image / 2 + 0.5).clamp(0, 1).float()


def preprocess_imagereward(images):
    """Resize bicubique 224, crop central et normalisation CLIP, tout en tenseurs."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    images = _float_images(images)
    device = images.device
    # Le backward du resize CUDA peut être non déterministe. Le transfert CPU garde
    # le graphe autograd et évite de désactiver le déterminisme de la baseline.
    resized = TF.resize(images.to("cpu"), 224, interpolation=InterpolationMode.BICUBIC, antialias=True)
    cropped = TF.center_crop(resized, [224, 224])
    normalized = TF.normalize(cropped, mean=[0.48145466, 0.4578275, 0.40821073],
                              std=[0.26862954, 0.26130258, 0.27577711])
    return normalized.to(device)


def image_reward(model, prompts, images):
    """Score officiel standardisé ImageReward, [B], avec gradient vers les images."""
    if isinstance(prompts, str):
        prompts = [prompts] * images.shape[0]
    if len(prompts) != images.shape[0]:
        raise ValueError("Un prompt est requis par image.")
    tokens = model.blip.tokenizer(prompts, padding="max_length", truncation=True,
                                  max_length=35, return_tensors="pt").to(images.device)
    prepared = preprocess_imagereward(images).to(dtype=next(model.parameters()).dtype)
    return model.score_gard(tokens.input_ids, tokens.attention_mask, prepared).reshape(-1)


def load_imagereward(device="cuda", cache_dir=None):
    """Architecture officielle, checkpoint et tokenizer épinglés ; fp32 et poids gelés."""
    import hashlib
    from pathlib import Path
    import warnings
    from huggingface_hub import hf_hub_download
    from transformers import BertTokenizer

    # L'import officiel modifie le filtre global des warnings : limiter cet effet à l'import.
    with warnings.catch_warnings():
        from ImageReward.ImageReward import ImageReward

    paths = {}
    for name in ("ImageReward.pt", "med_config.json"):
        paths[name] = hf_hub_download("zai-org/ImageReward", name,
                                      revision=IMAGEREWARD_MODEL_REVISION, cache_dir=cache_dir)
    digest = hashlib.sha256()
    with Path(paths["ImageReward.pt"]).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != IMAGEREWARD_WEIGHTS_SHA256:
        raise RuntimeError("Le SHA-256 des poids ImageReward ne correspond pas à la version figée.")

    # Le constructeur officiel initialise son tokenizer BERT habituel. On vérifie puis
    # remplace ce tokenizer par sa révision figée avant toute évaluation du score.
    model = ImageReward(med_config=paths["med_config.json"], device="cpu")
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased",
                                               revision=BERT_REVISION, cache_dir=cache_dir)
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    if tokenizer.get_vocab() != model.blip.tokenizer.get_vocab():
        raise RuntimeError("Le vocabulaire BERT officiel diffère de la révision figée.")
    model.blip.tokenizer = tokenizer
    state = torch.load(paths["ImageReward.pt"], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    model.device = torch.device(device)  # score_gard utilise cet attribut pour ses masques.
    return model


def check_latent_gradient(vae, latents, reward_fn, step_sizes=(0.01, 0.003, 0.001),
                          min_consecutive_steps=1):
    """Diagnostic fp32 : gradient de R(D(z)), dérivée directionnelle et ascension locale."""
    import math

    if not step_sizes or any(not math.isfinite(step) or step <= 0 for step in step_sizes):
        raise ValueError("Les pas de différence finie doivent être positifs et finis.")
    if any(first <= second for first, second in zip(step_sizes, step_sizes[1:])):
        raise ValueError("Les pas doivent être distincts et strictement décroissants.")
    if not isinstance(min_consecutive_steps, int) or not 1 <= min_consecutive_steps <= len(step_sizes):
        raise ValueError("Nombre de pas consécutifs requis invalide.")
    if latents.dtype != torch.float32 or next(vae.parameters()).dtype != torch.float32:
        raise ValueError("Ce test numérique nécessite les latents et le VAE en fp32.")
    if latents.shape[0] != 1:
        raise ValueError("Ce diagnostic traite une image à la fois.")
    with torch.enable_grad():
        clean_latent = latents.detach().clone().requires_grad_(True)
        image = decode_latents_differentiable(vae, clean_latent)
        reward = reward_fn(image)
        if reward.shape != (1,) or not torch.isfinite(reward).all():
            raise ValueError("La récompense doit renvoyer une valeur finie par image.")
        gradient = torch.autograd.grad(reward.sum(), clean_latent)[0]
    norm = gradient.norm().item()
    finite = bool(torch.isfinite(gradient).all()) and math.isfinite(norm)
    reward_value = reward.detach().item()
    saturation = ((image.detach() <= 0) | (image.detach() >= 1)).float().mean().item()
    del image, reward, clean_latent  # aucun graphe conservé entre les tests
    report = {
        "reward": reward_value, "gradient_shape": list(gradient.shape),
        "gradient_finite": finite, "gradient_l2": norm,
        "gradient_max_abs": gradient.abs().max().item(), "saturated_pixel_fraction": saturation,
        "vae_parameters_frozen": all(not p.requires_grad for p in vae.parameters()),
        "vae_parameter_grads_none": all(p.grad is None for p in vae.parameters()),
        "fd_atol": 1e-5, "fd_rtol": 0.10, "directional_checks": [],
        "step_sizes": list(step_sizes), "min_consecutive_steps": min_consecutive_steps,
        "finite_difference_mode": "enable_grad_with_leaf_requires_grad",
    }
    if finite and norm > 0:
        direction = gradient.detach() / norm
        # Même route que le backward ; un seul graphe temporaire à la fois.
        with torch.enable_grad():
            for step in step_sizes:
                values = []
                for sign in (1, -1):
                    point = (latents + sign * step * direction).detach().requires_grad_(True)
                    value = reward_fn(decode_latents_differentiable(vae, point))
                    if value.shape != (1,):
                        raise ValueError("La récompense doit renvoyer une valeur par image.")
                    values.append(value.detach().item())
                    del value, point
                plus, minus = values
                derivative = (plus - minus) / (2 * step)
                # Tolérance pratique fp32 du réseau complet, fixée avant le test.
                matches = abs(derivative - norm) <= 1e-5 + 0.10 * norm
                report["directional_checks"].append({
                    "step": step, "reward_plus": plus, "reward_minus": minus,
                    "finite_difference": derivative, "autograd_derivative": norm,
                    "forward_slope": (plus - reward_value) / step,
                    "backward_slope": (reward_value - minus) / step,
                    "finite": math.isfinite(plus) and math.isfinite(minus),
                    "derivative_matches": matches,
                    "local_ascent": plus > reward_value and minus < reward_value,
                })
    consecutive = longest = 0
    for check in report["directional_checks"]:
        consecutive = consecutive + 1 if check["derivative_matches"] and check["local_ascent"] else 0
        longest = max(longest, consecutive)
    report["max_consecutive_passing_steps"] = longest
    report["passed"] = (finite and norm > 0 and gradient.shape == latents.shape
                        and report["vae_parameters_frozen"] and report["vae_parameter_grads_none"]
                        and all(c["finite"] for c in report["directional_checks"])
                        and longest >= min_consecutive_steps)
    return report, gradient.detach()


@torch.no_grad()
def compare_imagereward_api(model, prompt, images):
    """Comparaison de valeurs uniquement : PIL est réservé à cette référence hors graphe."""
    import math
    from torchvision.transforms import functional as TF

    if images.shape[0] != 1:
        raise ValueError("La comparaison officielle traite une image à la fois.")
    # Une même image source, quantifiée explicitement pour l'API officielle PIL.
    pixels = (images[0].detach().cpu() * 255).round().clamp(0, 255).to(torch.uint8)
    pil_image = TF.to_pil_image(pixels)
    official_input = model.preprocess(pil_image).unsqueeze(0).to(model.device)
    tokens = model.blip.tokenizer(prompt, padding="max_length", truncation=True,
                                  max_length=35, return_tensors="pt").to(model.device)
    official_score = model.score(prompt, pil_image)
    same_input_score = model.score_gard(tokens.input_ids, tokens.attention_mask, official_input).item()
    tensor_score = image_reward(model, prompt, images).item()
    tensor_input = preprocess_imagereward(images)
    input_error = tensor_input - official_input
    exact_api_pass = abs(same_input_score - official_score) <= 1e-5 + 1e-5 * abs(official_score)
    return {
        "scores_finite": all(math.isfinite(v) for v in (official_score, same_input_score, tensor_score)),
        "official_pil_score": official_score, "score_gard_same_input": same_input_score,
        "tensor_differentiable_score": tensor_score,
        "api_same_input_absolute_error": abs(same_input_score - official_score),
        "api_same_input_passed": exact_api_pass, "api_atol": 1e-5, "api_rtol": 1e-5,
        "tensor_vs_pil_score_absolute_error": abs(tensor_score - official_score),
        "preprocess_max_abs_error": input_error.abs().max().item(),
        "preprocess_rmse": input_error.square().mean().sqrt().item(),
        "note": "PIL quantifie en 8 bits ; le resize tensoriel reste continu. Cet écart de prétraitement est un diagnostic, pas une égalité numérique promise.",
    }
