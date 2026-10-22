"""SD1.5 + DDIM déterministe : fonctions communes à Colab et Northflank."""

import time

import torch
from diffusers import DDIMScheduler


MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"


def configure_torch():
    """Fixe les choix numériques. CUBLAS_WORKSPACE_CONFIG doit précéder CUDA."""
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_pipeline(device="cuda"):
    """Charge les poids SD1.5 figés en fp16, avec leur scheduler converti en DDIM."""
    from diffusers import StableDiffusionPipeline

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Activer un runtime GPU dans Colab avant le chargement SD1.5.")
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        variant="fp16",
        torch_dtype=dtype,
        use_safetensors=True,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    # On conserve betas, steps_offset, set_alpha_to_one et clip_sample de SD1.5.
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    for model in (pipe.text_encoder, pipe.unet, pipe.vae):
        model.eval()
        model.requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def encode_prompt_cfg(pipe, prompt, negative_prompt=""):
    """Deux embeddings [1, 77, 768] pour SD1.5 : positif, puis négatif."""
    embeddings = []
    # Deux encodages séparés, comme encode_prompt dans la pipeline officielle.
    for text in (prompt, negative_prompt):
        tokens = pipe.tokenizer(
            text,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        attention_mask = None
        if getattr(pipe.text_encoder.config, "use_attention_mask", False):
            attention_mask = tokens.attention_mask.to(pipe._execution_device)
        embedding = pipe.text_encoder(
            tokens.input_ids.to(pipe._execution_device),
            attention_mask=attention_mask,
            return_dict=False,
        )[0]
        embeddings.append(embedding.to(dtype=pipe.text_encoder.dtype))
    return embeddings[0], embeddings[1]


def make_initial_noise(pipe, seed, height=512, width=512):
    """Bruit N(0,I) [1,4,64,64] en 512px, avant init_noise_sigma."""
    factor = pipe.vae_scale_factor
    if height % factor or width % factor:
        raise ValueError("La résolution doit être divisible par le facteur spatial du VAE.")
    shape = (1, pipe.unet.config.in_channels, height // factor, width // factor)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Tirage fp32 CPU puis transfert : même entrée sauvegardable sur T4 et A100.
    noise = torch.randn(shape, generator=generator, dtype=torch.float32)
    return noise.to(device=pipe._execution_device, dtype=pipe.unet.dtype)


def ddim_step(latents, noise_prediction, timestep, scheduler):
    """DDIM déterministe : epsilon-prediction et eta=0 uniquement."""
    if scheduler.num_inference_steps is None:
        raise ValueError("Appeler scheduler.set_timesteps avant ddim_step.")
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError("Cette baseline implémente seulement prediction_type='epsilon'.")
    if scheduler.config.thresholding:
        raise ValueError("Le thresholding dynamique ne fait pas partie de cette baseline.")
    if scheduler.config.timestep_spacing != "leading":
        raise ValueError("Cette baseline utilise uniquement les timesteps 'leading' de SD1.5.")

    timestep = int(timestep)
    previous_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    alpha_bar_t = scheduler.alphas_cumprod[timestep]
    if previous_timestep >= 0:
        alpha_bar_previous = scheduler.alphas_cumprod[previous_timestep]
    else:
        # SD1.5 : final_alpha_cumprod = alphas_cumprod[0], pas nécessairement 1.
        alpha_bar_previous = scheduler.final_alpha_cumprod

    # Scalaires fp32 du scheduler, sans arrondi préalable en fp16.
    alpha_t = alpha_bar_t ** 0.5
    sigma_t = (1 - alpha_bar_t) ** 0.5
    predicted_x0 = (latents - sigma_t * noise_prediction) / alpha_t
    if scheduler.config.clip_sample:
        limit = scheduler.config.clip_sample_range
        predicted_x0 = predicted_x0.clamp(-limit, limit)

    alpha_previous = alpha_bar_previous ** 0.5
    sigma_previous = (1 - alpha_bar_previous) ** 0.5
    previous_latents = alpha_previous * predicted_x0 + sigma_previous * noise_prediction
    return previous_latents, predicted_x0


@torch.no_grad()
def sample_ddim(
    pipe, initial_noise, prompt_embeds, negative_prompt_embeds,
    num_steps=50, guidance_scale=7.5, return_trace=False,
):
    """Boucle BASE lisible ; renvoie latents finaux et éventuelle trace CPU."""
    if guidance_scale <= 1:
        raise ValueError("Cette boucle suppose le CFG actif (guidance_scale > 1).")
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=initial_noise.device)
    latents = initial_noise.clone() * scheduler.init_noise_sigma
    # Ordre attendu par CFG : inconditionnel, puis conditionnel.
    cfg_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    trace = []

    for timestep in scheduler.timesteps:
        latent_input = torch.cat([latents, latents], dim=0)
        latent_input = scheduler.scale_model_input(latent_input, timestep)
        predicted_noise = pipe.unet(
            latent_input, timestep, encoder_hidden_states=cfg_embeddings,
            return_dict=False,
        )[0]
        noise_unconditional, noise_conditional = predicted_noise.chunk(2)
        # Équation (6) : epsilon_CFG = epsilon_u + w * (epsilon_c - epsilon_u).
        noise_cfg = noise_unconditional + guidance_scale * (noise_conditional - noise_unconditional)

        # Point d'insertion futur de la guidance de récompense ; aucune ici.
        latents, predicted_x0 = ddim_step(latents, noise_cfg, timestep, scheduler)
        latents = latents.detach()
        if not torch.isfinite(latents).all():
            raise FloatingPointError(f"Latents non finis au timestep {int(timestep)}.")
        if return_trace:
            trace.append(latents.detach().cpu().clone())
    return latents, trace


@torch.no_grad()
def decode_latents(pipe, latents):
    """Décode en tenseur image [B,3,H,W] dans [0,1], avant conversion PNG."""
    image = pipe.vae.decode(
        latents / pipe.vae.config.scaling_factor, return_dict=False,
    )[0]
    # Même ordre d'opérations et même dtype que VaeImageProcessor de Diffusers.
    return (image / 2 + 0.5).clamp(0, 1).float()


@torch.no_grad()
def sample_official(
    pipe, initial_noise, prompt_embeds, negative_prompt_embeds,
    num_steps=50, guidance_scale=7.5, return_trace=False,
):
    """Référence Diffusers, mêmes entrées explicites ; image brute + latents."""
    trace = []

    def record_step(pipeline, step_index, timestep, values):
        if return_trace:
            trace.append(values["latents"].detach().cpu().clone())
        return values

    images = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        latents=initial_noise.clone(),
        height=initial_noise.shape[-2] * pipe.vae_scale_factor,
        width=initial_noise.shape[-1] * pipe.vae_scale_factor,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        guidance_rescale=0.0,
        eta=0.0,
        output_type="pt",
        callback_on_step_end=record_step,
        callback_on_step_end_tensor_inputs=["latents"],
    ).images
    # La validation demande toujours la trace ; sinon pas de latents exposés.
    final_latents = trace[-1] if return_trace else None
    return final_latents, images.float(), trace


def measure_call(function, *args, **kwargs):
    """Temps synchronisé et pics CUDA ; ne mesure pas le chargement des poids."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    start = time.perf_counter()
    result = function(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    stats = {"seconds": time.perf_counter() - start}
    if torch.cuda.is_available():
        stats.update(
            allocated_before_mib=initial_memory / 2**20,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            additional_peak_mib=(torch.cuda.max_memory_allocated() - initial_memory) / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
        )
    return result, stats
