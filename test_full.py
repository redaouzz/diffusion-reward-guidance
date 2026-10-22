"""Tests CPU de FULL, sans téléchargement ni exigence de gain de récompense.

Les exemples analytiques vérifient le Jacobien ; les petits réseaux Diffusers
vérifient le câblage et la mémoire des graphes. Le pilote SD1.5 reste distinct.
"""

import gc
import math
from types import SimpleNamespace
import unittest
import weakref

import torch

from baseline import ddim_step, sample_ddim
from guidance_cov import cov_noise_prediction
from guidance_full import full_noise_prediction, sample_full
from rewards import decode_latents_differentiable, luminance_reward
from test_baseline import make_test_scheduler
from test_cov import LinearTestVAE, make_cov_test_pipe


class AffineTestUNet(torch.nn.Module):
    """Deux matrices connues, sélectionnées par l'embedding 0 ou 1 du CFG."""

    def __init__(self, matrix_u, matrix_c):
        super().__init__()
        self.matrices = torch.nn.Parameter(torch.stack([matrix_u, matrix_c]), requires_grad=False)
        self.eval()

    @property
    def dtype(self):
        return self.matrices.dtype

    def forward(self, latents, timestep, encoder_hidden_states, return_dict=False):
        conditional = encoder_hidden_states[:, 0, 0, None, None]
        matrices = self.matrices[0] + conditional * (self.matrices[1] - self.matrices[0])
        return (torch.einsum('bij,bjhw->bihw', matrices, latents),)


class TypedLinearTestVAE(LinearTestVAE):
    @property
    def dtype(self):
        return self.decoder.weight.dtype


def make_affine_pipe():
    matrix_u = torch.tensor([[.08, .03, 0, -.01], [.02, -.04, .015, .005],
                             [0, .02, .10, .01], [.01, .02, 0, .03]])
    matrix_c = torch.tensor([[.04, -.02, .03, .015], [-.01, .07, 0, -.02],
                             [.025, .01, -.03, .01], [.02, 0, .02, .09]])
    return SimpleNamespace(unet=AffineTestUNet(matrix_u, matrix_c),
                           vae=TypedLinearTestVAE(), scheduler=make_test_scheduler())


class FULLEquationTests(unittest.TestCase):
    def test_affine_cfg_jacobian_transpose_and_difference_from_cov(self):
        pipe = make_affine_pipe()
        pipe.scheduler.set_timesteps(50)
        timestep = pipe.scheduler.timesteps[25]
        alpha = pipe.scheduler.alphas_cumprod[timestep].sqrt()
        sigma = (1 - pipe.scheduler.alphas_cumprod[timestep]).sqrt()
        point = torch.linspace(-.03, .03, 16).reshape(1, 4, 2, 2)
        positive, negative = torch.ones(1, 1, 1), torch.zeros(1, 1, 1)
        weights = torch.tensor([.3, -.2, .7]).reshape(1, 3, 1, 1)
        reward_fn = lambda image: (image * weights).sum((1, 2, 3))
        beta, cfg = 1.7, 7.5
        actual_noise, gradient, report = full_noise_prediction(
            pipe, point, timestep, positive, negative, reward_fn, beta, cfg,
        )
        # Référence indépendante : D(mu)=mu[:3]/2+.5, sans saturation ici.
        v = torch.zeros_like(point)
        v[:, :3] = .5 * weights
        matrix_u, matrix_c = pipe.unet.matrices
        matrix_cfg = (1 - cfg) * matrix_u + cfg * matrix_c
        jacobian = (torch.eye(4) - sigma * matrix_cfg) / alpha
        expected_gradient = torch.einsum('ij,bihw->bjhw', jacobian, v)
        expected_noise = torch.einsum('ij,bjhw->bihw', matrix_cfg, point)
        torch.testing.assert_close(gradient, expected_gradient, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(actual_noise, expected_noise - sigma * beta * expected_gradient,
                                   rtol=2e-6, atol=2e-7)
        self.assertEqual(report['saturated_pixel_fraction'], 0)
        self.assertTrue(report['unet_graph_present'] and report['reward_connected'])
        self.assertFalse(torch.allclose(gradient, v / alpha))  # U-Net détaché : incorrect.
        self.assertFalse(torch.allclose(gradient, alpha * v))  # COV : autre Jacobien.
        cov_noise, _ = cov_noise_prediction(
            pipe.vae, point, expected_noise, timestep, pipe.scheduler, reward_fn, beta,
        )
        self.assertFalse(torch.allclose(actual_noise, cov_noise))
        self.assertFalse(gradient.requires_grad)
        self.assertIsNone(gradient.grad_fn)

    def test_gaussian_linear_reward_full_equals_cov_and_exact_gradient(self):
        pipe = make_affine_pipe()
        pipe.scheduler.set_timesteps(50)
        point = torch.linspace(-.1, .1, 16).reshape(1, 4, 2, 2)
        positive, negative = torch.ones(1, 1, 1), torch.zeros(1, 1, 1)
        beta = 2.3
        for timestep in pipe.scheduler.timesteps[[0, 24, 49]]:
            with self.subTest(timestep=int(timestep)):
                alpha = pipe.scheduler.alphas_cumprod[timestep].sqrt()
                sigma = (1 - pipe.scheduler.alphas_cumprod[timestep]).sqrt()
                # Prior N(0,I) : epsilon(z)=sigma*z et mu(z)=alpha*z.
                matrix = sigma * torch.eye(4)
                pipe.unet = AffineTestUNet(matrix, matrix)
                reward_fn = lambda image: image[:, 0].sum((1, 2))
                guided, gradient, _ = full_noise_prediction(
                    pipe, point, timestep, positive, negative, reward_fn, beta,
                )
                expected_v = torch.zeros_like(point)
                expected_v[:, 0] = .5
                expected_gradient = alpha * expected_v
                torch.testing.assert_close(gradient, expected_gradient, rtol=2e-4, atol=2e-7)
                cov_noise, _ = cov_noise_prediction(
                    pipe.vae, point, sigma * point, timestep, pipe.scheduler, reward_fn, beta,
                )
                torch.testing.assert_close(guided, cov_noise, rtol=2e-4, atol=1e-6)

    def test_noise_sign_and_ddim_displacement_including_final_step(self):
        pipe = make_affine_pipe()
        pipe.scheduler.set_timesteps(50)
        point = torch.full((1, 4, 2, 2), .01)
        positive, negative = torch.ones(1, 1, 1), torch.zeros(1, 1, 1)
        beta = .7
        for timestep in pipe.scheduler.timesteps[[20, 49]]:
            with self.subTest(timestep=int(timestep)):
                base_noise, _, _ = full_noise_prediction(
                    pipe, point, timestep, positive, negative, None, 0,
                )
                guided_noise, gradient, _ = full_noise_prediction(
                    pipe, point, timestep, positive, negative,
                    lambda image: image[:, 0].mean((1, 2)), beta,
                )
                alpha = pipe.scheduler.alphas_cumprod[timestep].sqrt()
                sigma = (1 - pipe.scheduler.alphas_cumprod[timestep]).sqrt()
                previous_timestep = int(timestep) - 20
                previous_ab = (pipe.scheduler.alphas_cumprod[previous_timestep]
                               if previous_timestep >= 0 else pipe.scheduler.final_alpha_cumprod)
                alpha_previous, sigma_previous = previous_ab.sqrt(), (1 - previous_ab).sqrt()
                expected_delta = (alpha_previous * sigma.square() / alpha - sigma_previous * sigma) * beta * gradient
                base_state, _ = ddim_step(point, base_noise, timestep, pipe.scheduler)
                guided_state, _ = ddim_step(point, guided_noise, timestep, pipe.scheduler)
                torch.testing.assert_close(guided_state - base_state, expected_delta, rtol=2e-4, atol=1e-8)
                torch.testing.assert_close(base_noise - guided_noise, sigma * beta * gradient,
                                           rtol=2e-6, atol=1e-8)


class FULLNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(2, cls.previous_threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_beta_zero_and_constants_preserve_every_baseline_state(self):
        for dtype in (torch.float32, torch.float16):
            pipe = make_cov_test_pipe(dtype)
            noise = torch.randn(1, 4, 8, 8, dtype=dtype)
            positive, negative = torch.randn(1, 4, 8, dtype=dtype), torch.randn(1, 4, 8, dtype=dtype)
            initial_copy = noise.clone()
            base, base_trace = sample_ddim(pipe, noise, positive, negative, num_steps=4, return_trace=True)

            def forbidden_reward(_image):
                self.fail('beta=0 ne doit pas calculer la récompense.')

            cases = [('beta_zero', forbidden_reward, 0, None),
                     ('connected_constant', lambda image: image.mean((1, 2, 3)) * 0 + 2, 3, True),
                     ('disconnected_constant', lambda image: image.new_full((1,), 2), 3, False)]
            for name, reward_fn, beta, connected in cases:
                with self.subTest(dtype=dtype, case=name):
                    actual, trace, reports = sample_full(
                        pipe, noise, positive, negative, reward_fn, beta,
                        num_steps=4, return_trace=True,
                    )
                    torch.testing.assert_close(actual, base, rtol=0, atol=0)
                    self.assertEqual(actual.dtype, dtype)
                    self.assertEqual(len(trace), 4)
                    for state, reference in zip(trace, base_trace):
                        torch.testing.assert_close(state, reference, rtol=0, atol=0)
                    for report in reports:
                        self.assertEqual(report['correction_l2_applied'], 0)
                        self.assertEqual(report['reward_connected'], connected)
                        self.assertEqual(report['unet_graph_present'], beta != 0)
                    self.assertTrue(torch.equal(noise, initial_copy))

    def test_real_fp32_directional_derivative(self):
        pipe = make_cov_test_pipe()
        pipe.scheduler.set_timesteps(50)
        timestep = pipe.scheduler.timesteps[35]
        point = .1 * torch.randn(1, 4, 8, 8)
        positive, negative = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
        _, gradient, _ = full_noise_prediction(
            pipe, point, timestep, positive, negative, luminance_reward, .5,
        )
        self.assertEqual(gradient.shape, point.shape)
        self.assertEqual(gradient.dtype, torch.float32)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.norm().item(), 0)
        direction = gradient / gradient.norm()
        expected_derivative = (gradient * direction).sum().item()
        alpha = pipe.scheduler.alphas_cumprod[timestep].sqrt()
        sigma = (1 - pipe.scheduler.alphas_cumprod[timestep]).sqrt()
        embeddings = torch.cat([negative, positive])

        @torch.no_grad()
        def reward_at(value):
            eps_u, eps_c = pipe.unet(torch.cat([value, value]), timestep,
                                    encoder_hidden_states=embeddings, return_dict=False)[0].chunk(2)
            mu = (value - sigma * (eps_u + 7.5 * (eps_c - eps_u))) / alpha
            return luminance_reward(decode_latents_differentiable(pipe.vae, mu)).item()

        differences = [(reward_at(point + h * direction) - reward_at(point - h * direction)) / (2 * h)
                       for h in (1e-3, 3e-4, 1e-4)]
        # On exige deux pas concordants, sans supposer un gain de trajectoire finale.
        agreements = [abs(value - expected_derivative) <= 1e-5 + .03 * abs(expected_derivative)
                      for value in differences]
        self.assertTrue(any(a and b for a, b in zip(agreements, agreements[1:])),
                        f'Autograd={expected_derivative}, différences finies={differences}')

    def test_checkpointing_preserves_real_gradient_in_eval(self):
        pipe = make_cov_test_pipe()
        pipe.scheduler.set_timesteps(50)
        timestep = pipe.scheduler.timesteps[25]
        point = .1 * torch.randn(1, 4, 8, 8)
        positive, negative = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
        before_noise, before_gradient, before_report = full_noise_prediction(
            pipe, point, timestep, positive, negative, luminance_reward, .3,
        )
        for model in (pipe.unet, pipe.vae):
            model.enable_gradient_checkpointing()
            self.assertTrue(model.is_gradient_checkpointing)
            self.assertFalse(model.training)
        noise, gradient, report = full_noise_prediction(
            pipe, point, timestep, positive, negative, luminance_reward, .3,
        )
        torch.testing.assert_close(noise, before_noise, rtol=0, atol=0)
        torch.testing.assert_close(gradient, before_gradient, rtol=0, atol=0)
        self.assertEqual(report['reward_at_mu'], before_report['reward_at_mu'])
        self.assertTrue(all(not p.requires_grad and p.grad is None
                            for model in (pipe.unet, pipe.vae) for p in model.parameters()))

    def test_unet_backward_frozen_weights_graph_release_and_reproducibility(self):
        pipe = make_cov_test_pipe()
        noise = (.1 * torch.randn(1, 4, 8, 8)).requires_grad_(True)
        positive = torch.randn(1, 4, 8, requires_grad=True)
        negative = torch.randn(1, 4, 8, requires_grad=True)
        graph_outputs, backward_norms, unet_calls = [], [], []

        def check_unet(_module, args, output):
            self.assertTrue(torch.is_grad_enabled() and args[0].requires_grad)
            self.assertTrue(output[0].requires_grad)
            self.assertIsNotNone(output[0].grad_fn)
            graph_outputs.append(weakref.ref(output[0]))
            args[0].register_hook(lambda gradient: backward_norms.append(gradient.detach().norm().item()))
            unet_calls.append(True)

        def check_decoder(_module, args, output):
            self.assertTrue(torch.is_grad_enabled() and args[0].requires_grad)
            self.assertEqual(args[0].dtype, torch.float32)
            graph_outputs.append(weakref.ref(output))

        unet_hook = pipe.unet.register_forward_hook(check_unet)
        vae_hook = pipe.vae.decoder.register_forward_hook(check_decoder)
        try:
            actual, trace, reports = sample_full(pipe, noise, positive, negative, luminance_reward,
                                                  .03, num_steps=4, return_trace=True)
            repeated, _, _ = sample_full(pipe, noise, positive, negative, luminance_reward, .03, num_steps=4)
        finally:
            unet_hook.remove()
            vae_hook.remove()
        self.assertEqual(len(unet_calls), 8)
        self.assertEqual(len(backward_norms), 8)
        self.assertTrue(all(math.isfinite(norm) for norm in backward_norms))
        self.assertTrue(any(norm > 0 for norm in backward_norms))
        torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
        for state in (actual, *trace):
            self.assertIsNone(state.grad_fn)
            self.assertFalse(state.requires_grad)
            self.assertTrue(torch.isfinite(state).all())
        for report in reports:
            self.assertEqual(report['status'], 'ok')
            self.assertTrue(report['unet_graph_present'] and report['state_detached'])
            self.assertTrue(report['reward_connected'] and report['finite'])
            self.assertTrue(all(not isinstance(value, torch.Tensor) for value in report.values()))
        self.assertTrue(all(not p.requires_grad and p.grad is None
                            for model in (pipe.unet, pipe.vae) for p in model.parameters()))
        self.assertIsNone(noise.grad)
        self.assertIsNone(positive.grad)
        self.assertIsNone(negative.grad)
        gc.collect()
        self.assertTrue(all(reference() is None for reference in graph_outputs))

    def test_nonfinite_reward_preserves_failure_and_invalid_models_are_rejected(self):
        pipe = make_cov_test_pipe()
        noise = torch.zeros(1, 4, 8, 8)
        positive, negative = torch.zeros(1, 4, 8), torch.zeros(1, 4, 8)
        diagnostics = []
        with self.assertRaises(FloatingPointError):
            sample_full(pipe, noise, positive, negative,
                        lambda image: image.new_tensor([float('nan')]), 1,
                        num_steps=3, diagnostics=diagnostics)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]['status'], 'error')
        self.assertEqual(diagnostics[0]['error_type'], 'FloatingPointError')
        for model in (pipe.unet, pipe.vae):
            parameter = next(model.parameters())
            parameter.requires_grad_(True)
            with self.assertRaises(ValueError):
                sample_full(pipe, noise, positive, negative, luminance_reward, 1, num_steps=3)
            parameter.requires_grad_(False)
            parameter.grad = torch.zeros_like(parameter)
            with self.assertRaises(ValueError):
                sample_full(pipe, noise, positive, negative, luminance_reward, 1, num_steps=3)
            parameter.grad = None

    def test_invalid_inputs_and_nonfinite_gradient_are_rejected(self):
        pipe = make_affine_pipe()
        pipe.scheduler.set_timesteps(50)
        timestep = pipe.scheduler.timesteps[25]
        point = torch.zeros(1, 4, 2, 2)
        positive, negative = torch.ones(1, 1, 1), torch.zeros(1, 1, 1)
        for beta in (-1, float('inf'), float('nan')):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                full_noise_prediction(pipe, point, timestep, positive, negative, luminance_reward, beta)
        for cfg in (1, float('inf'), float('nan')):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                full_noise_prediction(pipe, point, timestep, positive, negative, luminance_reward, 1, cfg)
        with self.assertRaises(FloatingPointError):
            full_noise_prediction(pipe, point + float('nan'), timestep, positive, negative, luminance_reward, 1)
        # Reward fini à zéro, dérivée infinie : contrôle distinct de la valeur du score.
        with self.assertRaises(FloatingPointError):
            full_noise_prediction(pipe, point, timestep, positive, negative,
                                  lambda image: (image.mean((1, 2, 3)) - .5).sqrt(), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
