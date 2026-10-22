"""FULL : gradient complet du surrogate de Tweedie, pas score cible exact."""

import math

import torch

from baseline import ddim_step
from rewards import decode_latents_differentiable


@torch.no_grad()
def full_noise_prediction(pipe, latents, timestep, prompt_embeds, negative_prompt_embeds,
                          reward_fn, beta, guidance_scale=7.5):
    """Renvoie (bruit corrigé, gradient non multiplié par beta, diagnostics).

    Le gradient retourné est détaché, de forme [1,C,H,W], ou None si beta=0.
    La feuille z est fp32 ; le cast vers le U-Net fp16 conserve autograd mais
    son backward reste fp16. Le VAE doit être fp32 et les réseaux gelés en eval.
    Un éventuel réseau dans reward_fn doit aussi être gelé par l'appelant.
    """
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta doit être fini et >= 0.")
    if not math.isfinite(guidance_scale) or guidance_scale <= 1:
        raise ValueError("CFG actif requis : guidance_scale fini > 1.")
    if latents.ndim != 4 or latents.shape[0] != 1:
        raise ValueError("Latents attendus : [1,C,H,W].")
    if latents.dtype != pipe.unet.dtype:
        raise ValueError("Le dtype des états DDIM doit rester celui du U-Net, comme BASE.")
    if pipe.vae.dtype != torch.float32:
        raise ValueError("Convertir explicitement le VAE en fp32 avant FULL.")
    for model in (pipe.unet, pipe.vae):
        if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ValueError("U-Net et VAE doivent être gelés, en eval(), sans gradients accumulés.")
    if not torch.isfinite(latents).all():
        raise FloatingPointError(f"Latents non finis au timestep {int(timestep)}.")

    scheduler = pipe.scheduler
    alpha_bar_t = scheduler.alphas_cumprod[int(timestep)].to(latents.device).float()
    alpha_t = alpha_bar_t.sqrt()
    sigma_t = (1 - alpha_bar_t).sqrt()
    report = dict(timestep=int(timestep), beta=float(beta), alpha=alpha_t.item(), sigma=sigma_t.item(),
                  reward_at_mu=None, reward_connected=None, gradient_l2=None, gradient_max_abs=None,
                  gradient_zero_fraction=None, guidance_l2=None, saturated_pixel_fraction=None,
                  unet_graph_present=False, correction_l2_fp32=0.0, correction_l2_applied=0.0,
                  relative_noise_correction_fp32=0.0, relative_noise_correction=0.0,
                  changed_noise_fraction=0.0, strong_correction=False, finite=True)
    cfg_embeddings = torch.cat([negative_prompt_embeds.detach(), prompt_embeds.detach()], dim=0)
    reward_gradient = None

    # Un graphe du pas courant seulement. beta=0 conserve l'inférence exacte BASE.
    with torch.set_grad_enabled(beta != 0), torch.autocast(device_type=latents.device.type, enabled=False):
        if beta == 0:
            noisy_latent = latents.detach()
        else:
            noisy_latent = latents.detach().float().clone().requires_grad_(True)
        # Le cast est différentiable : aucune branche du U-Net n'est détachée.
        unet_latent = noisy_latent.to(dtype=pipe.unet.dtype)
        latent_input = scheduler.scale_model_input(torch.cat([unet_latent, unet_latent]), timestep)
        predicted_noise = pipe.unet(
            latent_input, timestep, encoder_hidden_states=cfg_embeddings, return_dict=False,
        )[0]
        noise_unconditional, noise_conditional = predicted_noise.chunk(2)
        noise_cfg = noise_unconditional + guidance_scale * (noise_conditional - noise_unconditional)
        if noise_cfg.shape != latents.shape or not torch.isfinite(noise_cfg).all():
            raise FloatingPointError("Bruit CFG non fini ou de mauvaise forme.")
        report['unet_graph_present'] = noise_cfg.requires_grad and noise_cfg.grad_fn is not None
        report['noise_cfg_l2'] = noise_cfg.detach().float().norm().item()
        if beta != 0:
            if not report['unet_graph_present']:
                raise RuntimeError("Le graphe du U-Net manque : ce calcul ne serait pas FULL.")
            # mu brute en fp32, AVEC la dépendance epsilon_CFG(z_t).
            mu_t = (noisy_latent - sigma_t * noise_cfg.float()) / alpha_t
            image = decode_latents_differentiable(pipe.vae, mu_t)
            reward = reward_fn(image)
            if reward.shape != (1,) or not torch.isfinite(reward).all():
                raise FloatingPointError("reward_fn doit renvoyer un tenseur fini de forme [1].")
            if reward.requires_grad:
                # VJP direct : J_mu^T v. Aucun Jacobien dense, aucun retain_graph.
                reward_gradient = torch.autograd.grad(reward.sum(), noisy_latent, allow_unused=True)[0]
            report['reward_connected'] = reward_gradient is not None
            if reward_gradient is None:
                # Constante explicite ; le diagnostic signale aussi un detach accidentel du reward.
                reward_gradient = torch.zeros_like(noisy_latent)
            reward_gradient = reward_gradient.detach()
            if not torch.isfinite(reward_gradient).all():
                raise FloatingPointError(f"Gradient FULL non fini au timestep {int(timestep)}.")
            report.update(reward_at_mu=reward.detach().item(),
                          gradient_l2=reward_gradient.norm().item(),
                          gradient_max_abs=reward_gradient.abs().max().item(),
                          gradient_zero_fraction=(reward_gradient == 0).float().mean().item(),
                          saturated_pixel_fraction=((image.detach() <= 0) | (image.detach() >= 1)).float().mean().item())
            del reward, image, mu_t

    noise_cfg = noise_cfg.detach()
    del noisy_latent, unet_latent, latent_input, predicted_noise, noise_unconditional, noise_conditional
    if beta == 0:
        return noise_cfg, None, report

    # FP32 pour beta et la correction ; même dtype et même DDIM que BASE/COV ensuite.
    guidance = beta * reward_gradient
    correction = sigma_t * guidance
    noise_guided_fp32 = noise_cfg.float() - correction
    noise_guided = noise_guided_fp32.to(noise_cfg.dtype)
    if not torch.isfinite(noise_guided_fp32).all() or not torch.isfinite(noise_guided).all():
        raise FloatingPointError(f"Correction FULL non finie au timestep {int(timestep)}.")
    applied = noise_cfg.float() - noise_guided.float()
    relative_correction = applied.norm().item() / max(report['noise_cfg_l2'], 1e-12)
    report.update(guidance_l2=guidance.norm().item(),
                  correction_l2_fp32=correction.norm().item(), correction_l2_applied=applied.norm().item(),
                  relative_noise_correction_fp32=correction.norm().item() / max(report['noise_cfg_l2'], 1e-12),
                  relative_noise_correction=relative_correction,
                  changed_noise_fraction=(noise_guided != noise_cfg).float().mean().item(),
                  strong_correction=relative_correction > 0.1)
    if any(isinstance(value, float) and not math.isfinite(value) for value in report.values()):
        raise FloatingPointError(f"Norme/diagnostic non fini au timestep {int(timestep)}.")
    return noise_guided.detach(), reward_gradient, report


@torch.no_grad()
def sample_full(pipe, initial_noise, prompt_embeds, negative_prompt_embeds, reward_fn,
                beta, num_steps=50, guidance_scale=7.5, return_trace=False, diagnostics=None):
    """DDIM + FULL : (latents détachés, trace CPU, diagnostics scalaires par pas)."""
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta doit être fini et >= 0.")
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=initial_noise.device)
    latents = initial_noise.detach().clone() * scheduler.init_noise_sigma
    trace = []
    if diagnostics is None:
        diagnostics = []
    for step_index, timestep in enumerate(scheduler.timesteps):
        try:
            noise_guided, gradient, report = full_noise_prediction(
                pipe, latents, timestep, prompt_embeds, negative_prompt_embeds,
                reward_fn, beta, guidance_scale,
            )
            # Les DEUX occurrences epsilon du pas DDIM reçoivent le bruit corrigé.
            latents, predicted_x0 = ddim_step(latents, noise_guided, timestep, scheduler)
            latents = latents.detach()
            del noise_guided, gradient, predicted_x0
            if not torch.isfinite(latents).all():
                raise FloatingPointError(f"Latents FULL non finis au timestep {int(timestep)}.")
            report.update(step_index=step_index, state_detached=latents.grad_fn is None and not latents.requires_grad,
                          latent_l2=latents.float().norm().item(), status='ok')
            if latents.is_cuda:
                report['allocated_after_step_mib'] = torch.cuda.memory_allocated(latents.device) / 2**20
            diagnostics.append(report)
            if return_trace:
                trace.append(latents.cpu().clone())
        except Exception as error:
            diagnostics.append(dict(step_index=step_index, timestep=int(timestep), status='error',
                                    error_type=type(error).__name__, message=str(error)))
            raise
    if any(p.grad is not None for model in (pipe.unet, pipe.vae) for p in model.parameters()):
        raise RuntimeError("Un gradient de paramètre a été accumulé.")
    return latents, trace, diagnostics
