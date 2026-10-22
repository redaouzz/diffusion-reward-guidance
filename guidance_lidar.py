"""LiDAR : banque fixe, poids gaussiens et correction analytique."""

import math

import torch

from baseline import ddim_step


@torch.no_grad()
def lidar_guidance(latents, bank_latents, bank_rewards, alpha_t, sigma_t, beta):
    """Renvoie (g [1,C,H,W], diagnostics) depuis une banque détachée [n,C,H,W].

    Distances et softmax en FP64, guidance en FP32 (FP64 si latents FP64).
    Le centrage commun des logits/récompenses ne change pas les softmax.
    Aucune moyenne des distances, température ou normalisation n'est ajoutée.
    """
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta doit être fini et >= 0.")
    if latents.ndim != 4 or latents.shape[0] != 1 or latents.numel() == 0:
        raise ValueError("Latents attendus : [1,C,H,W], non vides.")
    if (bank_latents.ndim != 4 or bank_latents.shape[0] < 1
            or bank_latents.shape[1:] != latents.shape[1:]
            or bank_rewards.shape != (bank_latents.shape[0],)):
        raise ValueError("Banque attendue : [n,C,H,W] et récompenses [n], n >= 1.")
    for value in (latents, bank_latents, bank_rewards):
        if not value.is_floating_point() or value.device != latents.device:
            raise ValueError("Latents, banque et récompenses flottants sur le même device requis.")
        if not torch.isfinite(value).all():
            raise FloatingPointError("Latents, banque ou récompenses non finis.")
    if any(value.requires_grad or value.grad_fn is not None
           for value in (bank_latents, bank_rewards)):
        raise ValueError("Détacher la banque et ses récompenses avant la génération LiDAR.")
    alpha = torch.as_tensor(alpha_t, dtype=torch.float64, device=latents.device)
    sigma = torch.as_tensor(sigma_t, dtype=torch.float64, device=latents.device)
    if (alpha.numel() != 1 or sigma.numel() != 1 or not torch.isfinite(alpha).all()
            or not torch.isfinite(sigma).all() or alpha.item() < 0 or sigma.item() <= 0):
        raise ValueError("alpha doit être un scalaire fini >= 0 et sigma un scalaire fini > 0.")
    alpha, sigma = alpha.reshape(()), sigma.reshape(())
    sigma2 = sigma.square()
    bank = bank_latents.double()
    rewards = bank_rewards.double()

    # Équation (16) : sommer les carrés sur TOUTES les coordonnées du latent.
    differences = latents.double() - alpha * bank
    squared_distances = differences.square().flatten(1).sum(dim=1)
    log_likelihood = -squared_distances / (2 * sigma2)
    if not torch.isfinite(log_likelihood).all():
        raise FloatingPointError("Log-vraisemblances LiDAR non finies.")
    centered_logits = log_likelihood - log_likelihood.max()
    weights = torch.softmax(centered_logits, dim=0)
    constant_reward = bool((rewards == rewards[0]).all())
    identity_case = beta == 0 or constant_reward or bank.shape[0] == 1
    if identity_case:
        # Évite l'annulation numérique de deux softmax mathématiquement égaux.
        reward_logits = centered_logits
        reward_weights = weights.clone()
        guidance64 = torch.zeros_like(latents, dtype=torch.float64)
    else:
        reward_logits = centered_logits + beta * (rewards - rewards.max())
        if not torch.isfinite(reward_logits).all():
            raise FloatingPointError("Logits récompensés LiDAR non finis.")
        reward_weights = torch.softmax(reward_logits, dim=0)
        # Équation (17) : dérivée analytique du rapport des deux sommes.
        weight_difference = (reward_weights - weights).view(-1, 1, 1, 1)
        guidance64 = (alpha / sigma2) * (weight_difference * bank).sum(dim=0, keepdim=True)
    output_dtype = torch.float64 if latents.dtype == torch.float64 else torch.float32
    guidance = guidance64.to(dtype=output_dtype)
    if not torch.isfinite(guidance).all():
        raise FloatingPointError("Guidance LiDAR non finie après conversion numérique.")
    max_weight, max_reward_weight = weights.max().item(), reward_weights.max().item()
    report = dict(
        beta=float(beta), bank_size=bank.shape[0], alpha=alpha.item(), sigma=sigma.item(),
        weights=weights.cpu().tolist(), reward_weights=reward_weights.cpu().tolist(),
        ess=(1 / weights.square().sum()).item(), ess_reward=(1 / reward_weights.square().sum()).item(),
        log_likelihood_min=log_likelihood.min().item(), log_likelihood_max=log_likelihood.max().item(),
        reward_logits_min=reward_logits.min().item(), reward_logits_max=reward_logits.max().item(),
        max_weight=max_weight, max_reward_weight=max_reward_weight,
        near_one_hot=max_weight >= 1 - 1e-6, reward_near_one_hot=max_reward_weight >= 1 - 1e-6,
        exact_one_hot=bool((weights == 1).any() and (weights == 0).sum() == bank.shape[0] - 1),
        reward_exact_one_hot=bool((reward_weights == 1).any()
                                 and (reward_weights == 0).sum() == bank.shape[0] - 1),
        zero_weight_count=int((weights == 0).sum()), reward_zero_weight_count=int((reward_weights == 0).sum()),
        guidance_l2_fp64=guidance64.norm().item(), guidance_l2=guidance.double().norm().item(),
        guidance_max_abs=guidance.abs().max().item(),
        identity_case=identity_case, constant_reward=constant_reward, finite=True,
    )
    if any(isinstance(value, float) and not math.isfinite(value) for value in report.values()):
        raise FloatingPointError("Diagnostic LiDAR non fini.")
    return guidance.detach(), report


@torch.no_grad()
def lidar_noise_prediction(latents, noise_cfg, timestep, scheduler,
                           bank_latents, bank_rewards, beta):
    """epsilon_guided = epsilon_CFG - sigma*g ; même dtype de bruit que BASE."""
    if latents.shape != noise_cfg.shape or latents.device != noise_cfg.device:
        raise ValueError("Latents et bruit doivent avoir même forme et même device.")
    if not noise_cfg.is_floating_point() or not torch.isfinite(noise_cfg).all():
        raise FloatingPointError("Bruit CFG flottant et fini requis.")
    if not 0 <= int(timestep) < len(scheduler.alphas_cumprod):
        raise ValueError("Timestep hors de la configuration du scheduler.")
    # Même alpha_bar FP32 que COV/FULL ; sa racine est ensuite utilisée en FP64.
    alpha_bar_t = scheduler.alphas_cumprod[int(timestep)].to(latents.device).float()
    alpha_t, sigma_t = alpha_bar_t.sqrt(), (1 - alpha_bar_t).sqrt()
    guidance, report = lidar_guidance(latents, bank_latents, bank_rewards, alpha_t, sigma_t, beta)
    working_dtype = torch.float64 if noise_cfg.dtype == torch.float64 else torch.float32
    noise_working = noise_cfg.to(dtype=working_dtype)
    correction = sigma_t.to(dtype=working_dtype) * guidance.to(dtype=working_dtype)
    noise_guided_working = noise_working - correction
    # Les identités gardent directement le bruit d'origine, sans aller-retour.
    noise_guided = noise_cfg.detach() if report['identity_case'] else noise_guided_working.to(noise_cfg.dtype)
    if not torch.isfinite(noise_guided_working).all() or not torch.isfinite(noise_guided).all():
        raise FloatingPointError(f"Correction LiDAR non finie au timestep {int(timestep)}.")
    applied = noise_working - noise_guided.to(dtype=working_dtype)
    noise_norm = noise_working.double().norm().item()
    correction_norm = correction.double().norm().item()
    applied_norm = applied.double().norm().item()
    relative_correction = applied_norm / max(noise_norm, 1e-12)
    report.update(
        timestep=int(timestep), noise_cfg_l2=noise_norm,
        correction_l2_fp32=correction_norm, correction_l2_applied=applied_norm,
        relative_noise_correction_fp32=correction_norm / max(noise_norm, 1e-12),
        relative_noise_correction=relative_correction,
        changed_noise_fraction=(noise_guided != noise_cfg).float().mean().item(),
        strong_correction=relative_correction > 0.1,
    )
    if any(isinstance(value, float) and not math.isfinite(value) for value in report.values()):
        raise FloatingPointError("Norme de correction LiDAR non finie.")
    return noise_guided.detach(), report


@torch.no_grad()
def sample_lidar(pipe, initial_noise, prompt_embeds, negative_prompt_embeds,
                 bank_latents, bank_rewards, beta, num_steps=50, guidance_scale=7.5,
                 return_trace=False, diagnostics=None):
    """DDIM + LiDAR : (latents détachés, trace CPU, diagnostics par pas).

    Banque propre et récompenses sont calculées une fois par l'appelant.
    Aucun VAE, réseau de récompense ou backward n'est appelé dans cette boucle.
    Une liste diagnostics externe conserve les pas précédant un éventuel échec.
    """
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta doit être fini et >= 0.")
    if not math.isfinite(guidance_scale) or guidance_scale <= 1:
        raise ValueError("CFG actif requis : guidance_scale fini > 1.")
    for model in (pipe.unet, pipe.vae):
        if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ValueError("U-Net et VAE doivent être gelés, en eval(), sans gradients accumulés.")
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=initial_noise.device)
    latents = initial_noise.detach().clone() * scheduler.init_noise_sigma
    cfg_embeddings = torch.cat([negative_prompt_embeds.detach(), prompt_embeds.detach()], dim=0)
    trace = []
    if diagnostics is None:
        diagnostics = []
    for step_index, timestep in enumerate(scheduler.timesteps):
        try:
            latent_input = scheduler.scale_model_input(torch.cat([latents, latents]), timestep)
            predicted_noise = pipe.unet(
                latent_input, timestep, encoder_hidden_states=cfg_embeddings, return_dict=False,
            )[0]
            noise_unconditional, noise_conditional = predicted_noise.chunk(2)
            noise_cfg = noise_unconditional + guidance_scale * (noise_conditional - noise_unconditional)
            unet_graph_absent = predicted_noise.grad_fn is None and not predicted_noise.requires_grad
            del latent_input, predicted_noise, noise_unconditional, noise_conditional
            noise_guided, report = lidar_noise_prediction(
                latents, noise_cfg, timestep, scheduler, bank_latents, bank_rewards, beta,
            )
            # Le même bruit corrigé entre dans les DEUX termes du pas DDIM validé.
            latents, predicted_x0 = ddim_step(latents, noise_guided, timestep, scheduler)
            latents = latents.detach()
            del noise_cfg, noise_guided, predicted_x0
            if not torch.isfinite(latents).all():
                raise FloatingPointError(f"Latents LiDAR non finis au timestep {int(timestep)}.")
            report.update(step_index=step_index, unet_graph_absent=unet_graph_absent,
                          state_detached=latents.grad_fn is None and not latents.requires_grad,
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
