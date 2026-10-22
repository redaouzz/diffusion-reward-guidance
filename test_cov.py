"""Tests CPU de COV, sans téléchargement ni critère de gain de récompense.

Les petits réseaux vérifient les équations et le câblage autograd. La mémoire
et la génération SD1.5 complète sur T4 se mesurent dans le notebook 03.
"""

import gc
import math
from types import SimpleNamespace
import unittest
import weakref

import torch
from diffusers import UNet2DConditionModel

from baseline import ddim_step, sample_ddim
from guidance_cov import cov_noise_prediction, isotropic_guidance, sample_cov
from rewards import luminance_reward
from test_baseline import make_test_scheduler
from test_rewards import make_test_vae


class LinearTestVAE(torch.nn.Module):
    """Décodeur analytique de test : RGB = gain * les trois premiers canaux."""

    def __init__(self, gain=1.0):
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=1.0)
        self.decoder = torch.nn.Conv2d(4, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.decoder.weight.zero_()
            for channel in range(3):
                self.decoder.weight[channel, channel, 0, 0] = gain
        self.eval().requires_grad_(False)

    def decode(self, latents, return_dict=False):
        return (self.decoder(latents),)


def make_cov_test_pipe(dtype=torch.float32):
    """Vrai U-Net Diffusers aléatoire et petit VAE, tous deux gelés."""
    torch.manual_seed(23)
    unet = UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(8, 16),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=8, attention_head_dim=2, norm_num_groups=4,
    ).to(dtype=dtype).eval().requires_grad_(False)
    return SimpleNamespace(unet=unet, vae=make_test_vae(), scheduler=make_test_scheduler())


class COVEquationTests(unittest.TestCase):
    def test_gaussian_linear_reward_matches_exact_log_mgf_gradient(self):
        """La log-MGF du posterior fournit une référence analytique indépendante."""
        point = torch.tensor([0.2, -0.4, 0.7], dtype=torch.float64, requires_grad=True)
        direction = torch.tensor([0.3, 0.8, -0.2], dtype=torch.float64)
        beta = 2.3
        for alpha_value in (0.03, 0.6, 0.999):
            for tau2 in (0.2, 1.0, 3.0):
                with self.subTest(alpha=alpha_value, tau2=tau2):
                    alpha = point.new_tensor(alpha_value)
                    sigma = (1 - alpha.square()).sqrt()
                    posterior_variance = 1 / (1 / tau2 + alpha.square() / sigma.square())
                    posterior_mean = posterior_variance * alpha / sigma.square() * point
                    log_mgf = (beta * direction.dot(posterior_mean)
                               + 0.5 * beta**2 * posterior_variance * direction.square().sum())
                    exact_gradient = torch.autograd.grad(log_mgf, point)[0]
                    actual, kappa = isotropic_guidance(direction, alpha, sigma, beta, tau2)
                    torch.testing.assert_close(actual, exact_gradient, rtol=1e-12, atol=1e-12)
                    self.assertEqual(actual.shape, direction.shape)
                    self.assertTrue(torch.isfinite(actual).all())
                    if tau2 == 1:
                        torch.testing.assert_close(kappa, alpha, rtol=1e-12, atol=1e-12)
                        # Le détachement naïf beta / alpha est bien une autre méthode.
                        self.assertFalse(torch.allclose(actual, beta * direction / alpha))

    def test_noise_sign_gradient_and_ddim_displacement(self):
        vae = LinearTestVAE()
        scheduler = make_test_scheduler()
        scheduler.set_timesteps(50)
        beta, tau2 = 1.7, 0.4
        for timestep in (int(scheduler.timesteps[12]), int(scheduler.timesteps[-1])):
            with self.subTest(timestep=timestep):
                alpha = scheduler.alphas_cumprod[timestep].sqrt()
                sigma = (1 - scheduler.alphas_cumprod[timestep]).sqrt()
                noise = torch.full((1, 4, 2, 2), 0.2)
                latents = alpha * torch.full_like(noise, 0.1) + sigma * noise
                guided, report = cov_noise_prediction(
                    vae, latents, noise, timestep, scheduler,
                    lambda image: image[:, 0].mean((1, 2)), beta, tau2,
                )
                # D(mu) = (mu[:3] / 2 + .5), ici sans saturation.
                expected_v = torch.zeros_like(latents)
                expected_v[:, 0] = 0.5 / 4
                kappa = alpha * tau2 / (sigma.square() + alpha.square() * tau2)
                expected_g = beta * kappa * expected_v
                torch.testing.assert_close(guided, noise - sigma * expected_g, rtol=1e-6, atol=1e-7)
                self.assertLess(guided[0, 0, 0, 0].item(), noise[0, 0, 0, 0].item())
                self.assertAlmostEqual(report['gradient_l2'], expected_v.norm().item(), places=6)
                self.assertTrue(report['reward_connected'])
                base_next, _ = ddim_step(latents, noise, timestep, scheduler)
                cov_next, _ = ddim_step(latents, guided, timestep, scheduler)
                previous = timestep - scheduler.config.num_train_timesteps // 50
                alpha_bar_s = (scheduler.alphas_cumprod[previous] if previous >= 0
                               else scheduler.final_alpha_cumprod)
                alpha_s, sigma_s = alpha_bar_s.sqrt(), (1 - alpha_bar_s).sqrt()
                expected_delta = (alpha_s * sigma.square() / alpha - sigma_s * sigma) * expected_g
                torch.testing.assert_close(cov_next - base_next, expected_delta, rtol=2e-4, atol=1e-7)
                self.assertFalse(guided.requires_grad)

    def test_raw_mu_and_scheduler_clipping_remain_distinct(self):
        vae = LinearTestVAE(gain=0.1)
        for set_alpha_to_one in (False, True):
            scheduler = make_test_scheduler(clip_sample=True, set_alpha_to_one=set_alpha_to_one)
            scheduler.set_timesteps(50)
            for timestep in (int(scheduler.timesteps[0]), int(scheduler.timesteps[-1])):
                with self.subTest(final_alpha_one=set_alpha_to_one, timestep=timestep):
                    alpha = scheduler.alphas_cumprod[timestep].sqrt()
                    latents = torch.full((1, 4, 2, 2), 2.0) * alpha
                    noise = torch.zeros_like(latents)
                    decoder_inputs = []
                    hook = vae.decoder.register_forward_pre_hook(
                        lambda _model, args: decoder_inputs.append(args[0].detach().clone())
                    )
                    try:
                        guided, report = cov_noise_prediction(
                            vae, latents, noise, timestep, scheduler,
                            lambda image: image.mean((1, 2, 3)), beta=0.5,
                        )
                    finally:
                        hook.remove()
                    torch.testing.assert_close(decoder_inputs[0], torch.full_like(latents, 2.0))
                    self.assertAlmostEqual(report['reward_at_mu'], 0.6, places=6)
                    actual, predicted_x0 = ddim_step(latents, guided, timestep, scheduler)
                    expected = scheduler.step(guided, timestep, latents, eta=0.0)
                    torch.testing.assert_close(actual, expected.prev_sample, rtol=0, atol=0)
                    torch.testing.assert_close(predicted_x0, expected.pred_original_sample, rtol=0, atol=0)
                    self.assertLessEqual(predicted_x0.abs().max().item(), 1.0)

    def test_invalid_hyperparameters_are_rejected(self):
        for beta, tau2 in ((-1, 1), (float('nan'), 1), (float('inf'), 1),
                           (1, 0), (1, -1), (1, float('nan')), (1, float('inf'))):
            with self.subTest(beta=beta, tau2=tau2), self.assertRaises(ValueError):
                isotropic_guidance(torch.ones(2), torch.tensor(0.6), torch.tensor(0.8), beta, tau2)

    def test_nonfinite_inputs_rewards_gradients_and_cast_are_rejected(self):
        vae = LinearTestVAE()
        scheduler = make_test_scheduler()
        scheduler.set_timesteps(50)
        timestep = int(scheduler.timesteps[25])
        zero = torch.zeros(1, 4, 2, 2)
        for field in ('latents', 'noise'):
            for value in (float('nan'), float('inf')):
                with self.subTest(field=field, value=value), self.assertRaises(FloatingPointError):
                    bad = zero.clone()
                    bad[0, 0, 0, 0] = value
                    cov_noise_prediction(vae, bad if field == 'latents' else zero,
                                         bad if field == 'noise' else zero, timestep, scheduler,
                                         luminance_reward, beta=1)
        for reward_fn in (
            lambda image: image.new_tensor([float('nan')]),
            lambda image: image.new_zeros(2),
            # Valeur sqrt(0) finie, dérivée infinie : ne pas vérifier seulement R.
            lambda image: (image[:, 0, 0, 0] - 0.5).sqrt(),
        ):
            with self.assertRaises(FloatingPointError):
                cov_noise_prediction(vae, zero, zero, timestep, scheduler, reward_fn, beta=1)
        with self.assertRaises(FloatingPointError):
            cov_noise_prediction(vae, zero.half(), zero.half(), timestep, scheduler,
                                 lambda image: image[:, 0].mean((1, 2)), beta=1e8)


class COVSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(2, cls.previous_threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_beta_zero_and_constant_rewards_match_every_baseline_state(self):
        for dtype in (torch.float32, torch.float16):
            pipe = make_cov_test_pipe(dtype)
            noise = torch.randn(1, 4, 8, 8, dtype=dtype)
            positive = torch.randn(1, 4, 8, dtype=dtype)
            negative = torch.randn(1, 4, 8, dtype=dtype)
            initial_copy = noise.clone()
            base, base_trace = sample_ddim(pipe, noise, positive, negative, num_steps=4, return_trace=True)

            def forbidden_reward(_image):
                self.fail('beta=0 ne doit pas lancer le décodeur ou la récompense.')

            cases = (
                ('beta_zero', forbidden_reward, 0.0, None),
                ('connected_constant', lambda image: 0 * image.mean((1, 2, 3)) + 2, 3.0, True),
                ('disconnected_constant', lambda image: image.new_full((1,), 2), 3.0, False),
            )
            for name, reward_fn, beta, connected in cases:
                with self.subTest(dtype=dtype, case=name):
                    actual, trace, reports = sample_cov(
                        pipe, noise, positive, negative, reward_fn, beta,
                        num_steps=4, return_trace=True,
                    )
                    torch.testing.assert_close(actual, base, rtol=0, atol=0)
                    self.assertEqual(actual.dtype, dtype)
                    self.assertEqual(len(trace), len(base_trace))
                    for state, baseline_state in zip(trace, base_trace):
                        torch.testing.assert_close(state, baseline_state, rtol=0, atol=0)
                    for report in reports:
                        self.assertEqual(report['correction_l2_applied'], 0.0)
                        self.assertEqual(report['reward_connected'], connected)
                    self.assertTrue(torch.equal(noise, initial_copy))

    def test_real_network_gradients_graph_release_and_reproducibility(self):
        pipe = make_cov_test_pipe()
        noise = (0.1 * torch.randn(1, 4, 8, 8)).requires_grad_(True)
        positive = torch.randn(1, 4, 8, requires_grad=True)
        negative = torch.randn(1, 4, 8, requires_grad=True)
        decoder_graphs = []
        unet_calls = []

        def check_unet(_model, args, output):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(args[0].requires_grad)
            self.assertIsNone(output[0].grad_fn)
            self.assertFalse(output[0].requires_grad)
            unet_calls.append(True)

        def check_decoder(_model, args, output):
            self.assertTrue(torch.is_grad_enabled())
            self.assertTrue(args[0].requires_grad)
            self.assertEqual(args[0].dtype, torch.float32)
            decoder_graphs.append(weakref.ref(output))

        unet_hook = pipe.unet.register_forward_hook(check_unet)
        vae_hook = pipe.vae.decoder.register_forward_hook(check_decoder)
        try:
            actual, trace, reports = sample_cov(
                pipe, noise, positive, negative, luminance_reward, beta=0.3,
                num_steps=4, return_trace=True,
            )
            repeated, _, _ = sample_cov(
                pipe, noise, positive, negative, luminance_reward, beta=0.3, num_steps=4,
            )
        finally:
            unet_hook.remove()
            vae_hook.remove()
        self.assertEqual(len(unet_calls), 8)
        self.assertEqual(len(decoder_graphs), 8)
        torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
        for state in (actual, *trace):
            self.assertFalse(state.requires_grad)
            self.assertIsNone(state.grad_fn)
            self.assertTrue(torch.isfinite(state).all())
        for report in reports:
            self.assertEqual(report['status'], 'ok')
            self.assertTrue(report['unet_graph_absent'] and report['state_detached'])
            self.assertTrue(report['reward_connected'] and report['finite'])
            self.assertTrue(math.isfinite(report['gradient_l2']))
            self.assertGreaterEqual(report['gradient_l2'], 0)
            self.assertTrue(all(not isinstance(value, torch.Tensor) for value in report.values()))
        self.assertTrue(all(not p.requires_grad and p.grad is None
                            for model in (pipe.unet, pipe.vae) for p in model.parameters()))
        self.assertIsNone(noise.grad)
        self.assertIsNone(positive.grad)
        self.assertIsNone(negative.grad)
        gc.collect()
        self.assertTrue(all(reference() is None for reference in decoder_graphs))

    def test_failure_keeps_explicit_diagnostic_and_rejects_unfrozen_models(self):
        pipe = make_cov_test_pipe()
        noise = torch.zeros(1, 4, 8, 8)
        positive, negative = torch.zeros(1, 4, 8), torch.zeros(1, 4, 8)
        diagnostics = []
        with self.assertRaises(FloatingPointError):
            sample_cov(pipe, noise, positive, negative,
                       lambda image: image.new_tensor([float('nan')]), beta=1,
                       num_steps=3, diagnostics=diagnostics)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]['status'], 'error')
        self.assertEqual(diagnostics[0]['error_type'], 'FloatingPointError')
        for model in (pipe.unet, pipe.vae):
            parameter = next(model.parameters())
            parameter.requires_grad_(True)
            with self.assertRaises(ValueError):
                sample_cov(pipe, noise, positive, negative, luminance_reward, beta=1, num_steps=3)
            parameter.requires_grad_(False)
            parameter.grad = torch.zeros_like(parameter)
            with self.assertRaises(ValueError):
                sample_cov(pipe, noise, positive, negative, luminance_reward, beta=1, num_steps=3)
            parameter.grad = None


if __name__ == '__main__':
    unittest.main(verbosity=2)
