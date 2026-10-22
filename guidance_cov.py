"""COV : Jacobien de Tweedie isotrope, sans rétropropagation dans le U-Net."""

import math

import torch

from baseline import ddim_step
from rewards import decode_latents_differentiable


def isotropic_guidance(reward_gradient, alpha_t, sigma_t, beta, tau2=1.0):
    """g = beta * kappa * v ; approximation auxiliaire, pas score cible exact."""
    if not math.isfinite(beta) or beta < 0 or not math.isfinite(tau2) or tau2 <= 0:
        raise ValueError("beta doit être fini >= 0 et tau2 fini > 0.")
    kappa_t = alpha_t * tau2 / (sigma_t.square() + alpha_t.square() * tau2)
    guidance = beta * kappa_t * reward_gradient
    return guidance, kappa_t


def cov_noise_prediction(vae, latents, noise_cfg, timestep, scheduler,
                         reward_fn, beta, tau2=1.0):
    """Bruit corrigé et diagnostics scalaires. Gradient VAE/reward uniquement, batch 1.

    Le VAE doit être fp32 et gelé. reward_fn(images) renvoie [1]. Pour un réseau
    de récompense, l'appelant doit aussi geler ses poids et le placer en eval().
    Un reward indépendant des images est traité comme constant (diagnostic explicite).
    """
    if latents.shape != noise_cfg.shape or latents.ndim != 4 or latents.shape[0] != 1:
        raise ValueError("Latents et bruit de même forme [1,C,H,W] requis.")
    if next(vae.parameters()).dtype != torch.float32:
        raise ValueError("Convertir explicitement le VAE en fp32 avant COV.")
    if any(p.requires_grad for p in vae.parameters()) or vae.training:
        raise ValueError("Le VAE doit être gelé et en eval().")
    if not torch.isfinite(latents).all() or not torch.isfinite(noise_cfg).all():
        raise FloatingPointError(f"Entrée non finie au timestep {int(timestep)}.")

    alpha_bar_t = scheduler.alphas_cumprod[int(timestep)].to(latents.device).float()
    alpha_t = alpha_bar_t.sqrt()
    sigma_t = (1 - alpha_bar_t).sqrt()
    _, kappa_t = isotropic_guidance(torch.zeros((), device=latents.device),
                                   alpha_t, sigma_t, beta, tau2)
    report = dict(timestep=int(timestep), alpha=alpha_t.item(), sigma=sigma_t.item(),
                  kappa=kappa_t.item(), beta=float(beta), tau2=float(tau2),
                  reward_at_mu=None, gradient_l2=None, gradient_max_abs=None,
                  reward_connected=None, saturated_pixel_fraction=None,
                  noise_cfg_l2=noise_cfg.float().norm().item(), guidance_l2=None,
                  correction_l2_fp32=0.0, correction_l2_applied=0.0,
                  relative_noise_correction_fp32=0.0,
                  relative_noise_correction=0.0, changed_noise_fraction=0.0,
                  strong_correction=False, finite=True)
    if beta == 0:
        # Identité BASE et aucun décodage/backward inutile ; gradient non mesuré.
        return noise_cfg.detach(), report

    # mu brute : ni clipping scheduler, ni graphe vers z_t / epsilon_CFG.
    # L'autocast externe éventuel ne doit pas réduire la précision du gradient.
    with torch.enable_grad(), torch.autocast(device_type=latents.device.type, enabled=False):
        mu_t = (latents.detach().float() - sigma_t * noise_cfg.detach().float()) / alpha_t
        clean_latent = mu_t.detach().clone().requires_grad_(True)
        image = decode_latents_differentiable(vae, clean_latent)
        reward = reward_fn(image)
        if reward.shape != (1,) or not torch.isfinite(reward).all():
            raise FloatingPointError("reward_fn doit renvoyer un tenseur fini de forme [1].")
        reward_gradient = None
        if reward.requires_grad:
            reward_gradient = torch.autograd.grad(
                reward.sum(), clean_latent, allow_unused=True,
            )[0]
        report['reward_connected'] = reward_gradient is not None
        if reward_gradient is None:
            reward_gradient = torch.zeros_like(clean_latent)

    # create_graph=False : les sorties ne retiennent aucun graphe de backward.
    reward_gradient = reward_gradient.detach()
    if not torch.isfinite(reward_gradient).all():
        raise FloatingPointError(f"Gradient non fini au timestep {int(timestep)}.")
    guidance, _ = isotropic_guidance(reward_gradient, alpha_t, sigma_t, beta, tau2)
    correction = sigma_t * guidance
    noise_guided_fp32 = noise_cfg.detach().float() - correction
    # Même dtype de bruit et même arithmétique DDIM que BASE. La quantification
    # de cette correction est mesurée ; aucun clipping/normalisation de g ajouté.
    noise_guided = noise_guided_fp32.to(noise_cfg.dtype)
    if not torch.isfinite(noise_guided_fp32).all() or not torch.isfinite(noise_guided).all():
        raise FloatingPointError(f"Correction non finie au timestep {int(timestep)}.")

    applied = noise_cfg.detach().float() - noise_guided.float()
    relative_correction = applied.norm().item() / max(noise_cfg.float().norm().item(), 1e-12)
    report.update(
        reward_at_mu=reward.detach().item(), gradient_l2=reward_gradient.norm().item(),
        guidance_l2=guidance.norm().item(),
        gradient_max_abs=reward_gradient.abs().max().item(),
        saturated_pixel_fraction=((image.detach() <= 0) | (image.detach() >= 1)).float().mean().item(),
        correction_l2_fp32=correction.norm().item(), correction_l2_applied=applied.norm().item(),
        relative_noise_correction_fp32=correction.norm().item() / max(report['noise_cfg_l2'], 1e-12),
        relative_noise_correction=relative_correction,
        changed_noise_fraction=(noise_guided != noise_cfg).float().mean().item(),
        strong_correction=relative_correction > 0.1,
    )
    if any(isinstance(value, float) and not math.isfinite(value) for value in report.values()):
        raise FloatingPointError(f"Norme/diagnostic non fini au timestep {int(timestep)}.")
    return noise_guided.detach(), report


@torch.no_grad()
def sample_cov(pipe, initial_noise, prompt_embeds, negative_prompt_embeds, reward_fn,
               beta, tau2=1.0, num_steps=50, guidance_scale=7.5,
               return_trace=False, diagnostics=None):
    """DDIM + COV ; renvoie (latents détachés, trace CPU, diagnostics par pas).

    diagnostics peut être une liste externe pour conserver les pas précédant un
    échec. Le seul bloc différentiable est dans cov_noise_prediction.
    """
    if not math.isfinite(beta) or beta < 0 or not math.isfinite(tau2) or tau2 <= 0:
        raise ValueError("beta doit être fini >= 0 et tau2 fini > 0.")
    if guidance_scale <= 1:
        raise ValueError("Cette boucle suppose le CFG actif (guidance_scale > 1).")
    for model in (pipe.unet, pipe.vae):
        if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ValueError("U-Net et VAE doivent être gelés, en eval(), sans gradients accumulés.")
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=initial_noise.device)
    latents = initial_noise.detach().clone() * scheduler.init_noise_sigma
    cfg_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    trace = []
    if diagnostics is None:
        diagnostics = []

    for step_index, timestep in enumerate(scheduler.timesteps):
        try:
            latent_input = torch.cat([latents, latents], dim=0)
            latent_input = scheduler.scale_model_input(latent_input, timestep)
            # Identique à BASE ; @no_grad empêche la construction du graphe U-Net.
            predicted_noise = pipe.unet(
                latent_input, timestep, encoder_hidden_states=cfg_embeddings,
                return_dict=False,
            )[0]
            noise_unconditional, noise_conditional = predicted_noise.chunk(2)
            noise_cfg = noise_unconditional + guidance_scale * (noise_conditional - noise_unconditional)
            unet_graph_absent = predicted_noise.grad_fn is None and not predicted_noise.requires_grad
            del latent_input, predicted_noise, noise_unconditional, noise_conditional

            noise_guided, report = cov_noise_prediction(
                pipe.vae, latents, noise_cfg, timestep, scheduler, reward_fn, beta, tau2,
            )
            # Le même bruit corrigé entre dans les DEUX occurrences du pas DDIM.
            # ddim_step seul applique l'éventuel clipping et la convention finale.
            latents, predicted_x0 = ddim_step(latents, noise_guided, timestep, scheduler)
            latents = latents.detach()
            del noise_cfg, noise_guided, predicted_x0
            if not torch.isfinite(latents).all():
                raise FloatingPointError(f"Latents non finis au timestep {int(timestep)}.")
            report.update(step_index=step_index, unet_graph_absent=unet_graph_absent,
                          state_detached=latents.grad_fn is None and not latents.requires_grad,
                          latent_l2=latents.float().norm().item(), status="ok")
            if latents.is_cuda:
                report['allocated_after_step_mib'] = torch.cuda.memory_allocated(latents.device) / 2**20
            diagnostics.append(report)
            if return_trace:
                trace.append(latents.cpu().clone())
        except Exception as error:
            diagnostics.append(dict(step_index=step_index, timestep=int(timestep), status="error",
                                    error_type=type(error).__name__, message=str(error)))
            raise

    if any(p.grad is not None for model in (pipe.unet, pipe.vae) for p in model.parameters()):
        raise RuntimeError("Un gradient de paramètre a été accumulé.")
    return latents, trace, diagnostics
