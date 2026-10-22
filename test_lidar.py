"""Tests CPU de LiDAR : équations, stabilité et intégration DDIM sans backward.

Les réseaux Diffusers sont petits et aléatoires, sans téléchargement. Ces tests
ne mesurent ni la qualité de SD1.5 ni la mémoire ou le temps réels sur T4.
"""

import json
import math
import unittest
from unittest.mock import patch

import torch

from baseline import ddim_step, sample_ddim
from guidance_lidar import lidar_guidance, lidar_noise_prediction, sample_lidar
from test_baseline import make_test_scheduler
from test_cov import make_cov_test_pipe


class LiDAREquationTests(unittest.TestCase):
    def test_analytic_gradient_matches_autograd_logsumexp_and_finite_differences(self):
        """Référence indépendante : dériver log(sum w_i exp(beta*r_i))."""
        point = torch.tensor([.2, -.3, .1, .4]).reshape(1, 1, 2, 2).requires_grad_(True)
        bank = torch.tensor([[-.4, .2, .1, .5], [.3, -.2, .7, -.1], [.5, .6, -.2, .3]])
        bank = bank.reshape(3, 1, 2, 2)
        rewards = torch.tensor([-.25, .5, -.125])
        alpha, sigma, beta = .6, .8, 1.7

        def log_estimator(state):
            logits = -(state - alpha * bank).flatten(1).square().sum(1) / (2 * sigma**2)
            return torch.logsumexp(logits + beta * rewards, 0) - torch.logsumexp(logits, 0)

        expected = torch.autograd.grad(log_estimator(point), point)[0]
        actual, report = lidar_guidance(point, bank, rewards, alpha, sigma, beta)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)
        self.assertEqual(actual.shape, point.shape)
        self.assertFalse(actual.requires_grad)
        self.assertTrue(report['finite'])
        # Une direction fixée, distincte du gradient, vérifie aussi la dérivée.
        direction = torch.tensor([.3, -.2, .5, .1]).reshape_as(point)
        direction = direction / direction.norm()
        derivative = (actual * direction).sum().item()
        for step in (.01, .003):
            finite_difference = ((log_estimator(point.detach() + step * direction)
                                  - log_estimator(point.detach() - step * direction)) / (2 * step)).item()
            self.assertTrue(math.isclose(finite_difference, derivative, rel_tol=.003, abs_tol=1e-5))

    def test_distances_sum_all_dimensions_without_mean_normalization(self):
        point = torch.full((1, 1, 4, 4), .25, dtype=torch.float64)
        bank = torch.stack([torch.zeros_like(point[0]), torch.ones_like(point[0])])
        rewards = torch.tensor([0., 1.], dtype=torch.float64)
        alpha, sigma, beta = .8, .6, .7
        actual, report = lidar_guidance(point, bank, rewards, alpha, sigma, beta)
        # Deux distances connues analytiquement, avec d=16 et non une moyenne.
        distances = torch.tensor([16 * .25**2, 16 * (.25 - alpha)**2], dtype=torch.float64)
        logits = -distances / (2 * sigma**2)
        weights = logits.softmax(0)
        reward_weights = (logits + beta * rewards).softmax(0)
        torch.testing.assert_close(torch.tensor(report['weights'], dtype=torch.float64), weights,
                                   rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(torch.tensor(report['reward_weights'], dtype=torch.float64), reward_weights,
                                   rtol=1e-12, atol=1e-12)
        expected = torch.full_like(point, alpha / sigma**2 * (reward_weights[1] - weights[1]))
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        wrong_mean_weights = (logits / point.numel()).softmax(0)
        self.assertGreater((weights - wrong_mean_weights).abs().max().item(), .1)

    def test_zero_beta_constant_reward_and_singleton_give_exact_zero(self):
        for dtype in (torch.float16, torch.float32, torch.float64):
            point = torch.tensor([.2, -.1, .4, -.3], dtype=dtype).reshape(1, 1, 2, 2)
            bank = torch.cat([point + .4, point - .2, -point])
            rewards = torch.tensor([0., .25, .75], dtype=dtype)
            cases = ((bank, rewards, 0.), (bank, torch.full_like(rewards, 3.), 1e6),
                     (bank[:1], rewards[:1], 1e6))
            for candidates, scores, beta in cases:
                with self.subTest(dtype=dtype, count=len(candidates), beta=beta):
                    actual, report = lidar_guidance(point, candidates, scores, .6, .8, beta)
                    self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))
                    self.assertEqual(actual.dtype, torch.float64 if dtype == torch.float64 else torch.float32)
                    self.assertTrue(report['finite'])

    def test_normalized_weights_ess_and_reward_concentration(self):
        """Ici L est constant : ESS décroît avec beta ; ce n'est pas une loi générale."""
        point = torch.zeros(1, 1, 1, 2)
        bank = torch.tensor([[-1., 0.], [0., 1.], [1., 0.]]).reshape(3, 1, 1, 2)
        rewards = torch.tensor([0., .5, 1.])
        previous_ess = 3.0
        for beta in (0., .5, 10., 1000.):
            actual, report = lidar_guidance(point, bank, rewards, .6, .8, beta)
            weights = torch.tensor(report['weights'], dtype=torch.float64)
            reward_weights = torch.tensor(report['reward_weights'], dtype=torch.float64)
            for distribution, key in ((weights, 'ess'), (reward_weights, 'ess_reward')):
                self.assertAlmostEqual(distribution.sum().item(), 1., places=12)
                self.assertTrue((distribution >= 0).all())
                self.assertAlmostEqual(report[key], 1 / distribution.square().sum().item(), places=10)
                self.assertGreaterEqual(report[key], 1. - 1e-12)
                self.assertLessEqual(report[key], len(bank) + 1e-12)
            self.assertAlmostEqual(report['ess'], 3., places=10)
            self.assertLessEqual(report['ess_reward'], previous_ess + 1e-12)
            previous_ess = report['ess_reward']
            self.assertTrue(torch.isfinite(actual).all())
        self.assertAlmostEqual(previous_ess, 1., places=10)
        self.assertGreater(reward_weights[-1].item(), .999999)

    def test_large_beta_and_reward_shift_remain_finite_and_invariant(self):
        point = torch.tensor([.2, -.3], dtype=torch.float64).reshape(1, 1, 1, 2)
        bank = torch.tensor([[-.5, .25], [.25, .5], [.75, -.25]], dtype=torch.float64).reshape(3, 1, 1, 2)
        rewards = torch.tensor([-.5, -.25, 0.], dtype=torch.float64)
        for beta in (1e-4, 1., 100., 1e6, 1e300):
            with self.subTest(beta=beta):
                actual, report = lidar_guidance(point, bank, rewards, .6, .8, beta)
                shifted, shifted_report = lidar_guidance(point, bank, rewards + 2**20, .6, .8, beta)
                torch.testing.assert_close(actual, shifted, rtol=1e-10, atol=1e-12)
                self.assertEqual(report['reward_weights'], shifted_report['reward_weights'])
                self.assertTrue(torch.isfinite(actual).all())
                json.dumps(report, allow_nan=False)
        # Des carrés qui déborderaient en fp16 sont calculés en précision supérieure.
        half_point = torch.full((1, 1, 2, 2), 2000., dtype=torch.float16)
        half_bank = torch.cat([half_point - 2, half_point + 2])
        actual, report = lidar_guidance(half_point, half_bank, torch.tensor([0., 1.]), .6, .8, 2.)
        self.assertTrue(torch.isfinite(actual).all())
        json.dumps(report, allow_nan=False)

    def test_guided_noise_sign_and_both_ddim_terms_including_final_step(self):
        scheduler = make_test_scheduler()
        scheduler.set_timesteps(50)
        point = torch.zeros(1, 4, 2, 2)
        noise = torch.full_like(point, .1)
        bank = torch.zeros(2, 4, 2, 2)
        bank[0, 0] = -.02
        bank[1, 0] = .02
        rewards = torch.tensor([0., 1.])
        beta = .02
        for timestep in (int(scheduler.timesteps[24]), int(scheduler.timesteps[-1])):
            with self.subTest(timestep=timestep):
                alpha = scheduler.alphas_cumprod[timestep].sqrt()
                sigma = (1 - scheduler.alphas_cumprod[timestep]).sqrt()
                guided, report = lidar_noise_prediction(point, noise, timestep, scheduler, bank, rewards, beta)
                # Les deux candidats ont exactement la même distance à z=0.
                rewarded_positive_weight = 1 / (1 + math.exp(-beta))
                expected_g = torch.zeros_like(point)
                expected_g[:, 0] = alpha / sigma.square() * .04 * (rewarded_positive_weight - .5)
                torch.testing.assert_close(guided, noise - sigma * expected_g, rtol=1e-6, atol=1e-7)
                self.assertLess(guided[0, 0, 0, 0].item(), noise[0, 0, 0, 0].item())
                base_next, _ = ddim_step(point, noise, timestep, scheduler)
                guided_next, _ = ddim_step(point, guided, timestep, scheduler)
                previous = timestep - scheduler.config.num_train_timesteps // 50
                alpha_bar_previous = (scheduler.alphas_cumprod[previous] if previous >= 0
                                      else scheduler.final_alpha_cumprod)
                alpha_previous = alpha_bar_previous.sqrt()
                sigma_previous = (1 - alpha_bar_previous).sqrt()
                expected_delta = (alpha_previous * sigma.square() / alpha - sigma_previous * sigma) * expected_g
                torch.testing.assert_close(guided_next - base_next, expected_delta, rtol=3e-4, atol=1e-7)
                expected_step = scheduler.step(guided, timestep, point, eta=0.)
                torch.testing.assert_close(guided_next, expected_step.prev_sample, rtol=0, atol=0)
                self.assertEqual(guided.shape, point.shape)
                self.assertTrue(report['finite'])

    def test_invalid_inputs_are_rejected(self):
        point = torch.zeros(1, 1, 2, 2)
        bank = torch.cat([point, point + 1])
        rewards = torch.tensor([0., 1.])
        for beta in (-1., float('nan'), float('inf')):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                lidar_guidance(point, bank, rewards, .6, .8, beta)
        for alpha, sigma in ((-.1, .8), (.6, 0.), (.6, -.1), (float('nan'), .8), (.6, float('inf'))):
            with self.subTest(alpha=alpha, sigma=sigma), self.assertRaises(ValueError):
                lidar_guidance(point, bank, rewards, alpha, sigma, 1.)
        for bad_bank, bad_rewards in ((bank, rewards[:, None]), (bank, rewards[:1]),
                                      (bank[:0], rewards[:0]), (bank[:, :, :, :1], rewards),
                                      (bank.clone().requires_grad_(True), rewards),
                                      (bank, rewards.clone().requires_grad_(True))):
            with self.assertRaises(ValueError):
                lidar_guidance(point, bad_bank, bad_rewards, .6, .8, 1.)
        for value in (float('nan'), float('inf')):
            for field in ('latents', 'bank', 'rewards'):
                with self.subTest(value=value, field=field), self.assertRaises(FloatingPointError):
                    lidar_guidance(point + value if field == 'latents' else point,
                                   bank + value if field == 'bank' else bank,
                                   rewards + value if field == 'rewards' else rewards, .6, .8, 1.)


class LiDARSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(2, cls.previous_threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_zero_beta_constant_rewards_and_singleton_match_every_baseline_state(self):
        for dtype in (torch.float32, torch.float16):
            pipe = make_cov_test_pipe(dtype)
            noise = torch.randn(1, 4, 8, 8, dtype=dtype)
            positive = torch.randn(1, 4, 8, dtype=dtype)
            negative = torch.randn(1, 4, 8, dtype=dtype)
            bank = torch.randn(3, 4, 8, 8, dtype=dtype)
            rewards = torch.tensor([0., .25, .75])
            base, base_trace = sample_ddim(pipe, noise, positive, negative, num_steps=4, return_trace=True)
            cases = ((bank, rewards, 0.), (bank, torch.ones(3), 3.), (bank[:1], rewards[:1], 3.))
            for candidates, scores, beta in cases:
                with self.subTest(dtype=dtype, count=len(candidates), beta=beta):
                    actual, trace, reports = sample_lidar(
                        pipe, noise, positive, negative, candidates, scores, beta,
                        num_steps=4, return_trace=True,
                    )
                    torch.testing.assert_close(actual, base, rtol=0, atol=0)
                    self.assertEqual(actual.dtype, dtype)
                    self.assertEqual(len(trace), len(base_trace))
                    for state, reference in zip(trace, base_trace):
                        torch.testing.assert_close(state, reference, rtol=0, atol=0)
                    self.assertTrue(all(row['correction_l2_applied'] == 0 for row in reports))

    def test_sampling_never_decodes_or_backpropagates_and_is_reproducible(self):
        pipe = make_cov_test_pipe()
        noise = (.1 * torch.randn(1, 4, 8, 8)).requires_grad_(True)
        positive = torch.randn(1, 4, 8, requires_grad=True)
        negative = torch.randn(1, 4, 8, requires_grad=True)
        bank = .1 * torch.randn(3, 4, 8, 8)
        rewards = torch.tensor([-.25, 0., .5])
        copies = [value.detach().clone() for value in (noise, positive, negative, bank, rewards)]
        unet_calls = []

        def check_unet(_model, args, output):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(args[0].requires_grad)
            self.assertFalse(output[0].requires_grad)
            self.assertIsNone(output[0].grad_fn)
            unet_calls.append(True)

        hook = pipe.unet.register_forward_hook(check_unet)
        try:
            with patch.object(pipe.vae, 'decode', side_effect=AssertionError('VAE interdit dans LiDAR.')), \
                 patch('torch.autograd.grad', side_effect=AssertionError('Backward interdit dans LiDAR.')):
                actual, trace, reports = sample_lidar(
                    pipe, noise, positive, negative, bank, rewards, .3, num_steps=4, return_trace=True,
                )
                repeated, _, _ = sample_lidar(pipe, noise, positive, negative, bank, rewards, .3, num_steps=4)
        finally:
            hook.remove()
        self.assertEqual(len(unet_calls), 8)
        torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
        for state in (actual, *trace):
            self.assertFalse(state.requires_grad)
            self.assertIsNone(state.grad_fn)
            self.assertTrue(torch.isfinite(state).all())
        for report in reports:
            self.assertEqual(report['status'], 'ok')
            self.assertTrue(report['unet_graph_absent'] and report['state_detached'] and report['finite'])
            json.dumps(report, allow_nan=False)
        self.assertTrue(any(row['correction_l2_applied'] > 0 for row in reports))
        for value, saved in zip((noise, positive, negative, bank, rewards), copies):
            torch.testing.assert_close(value.detach(), saved, rtol=0, atol=0)
            self.assertIsNone(value.grad)
        self.assertTrue(all(not p.requires_grad and p.grad is None
                            for model in (pipe.unet, pipe.vae) for p in model.parameters()))

    def test_runtime_failure_keeps_diagnostics_and_invalid_models_are_rejected(self):
        pipe = make_cov_test_pipe()
        noise = torch.zeros(1, 4, 8, 8)
        positive, negative = torch.zeros(1, 4, 8), torch.zeros(1, 4, 8)
        bank = torch.cat([noise, noise + .1])
        rewards = torch.tensor([0., 1.])
        diagnostics = []
        hook = pipe.unet.register_forward_hook(lambda _model, _args, output: (output[0] * float('nan'),))
        try:
            with self.assertRaises(FloatingPointError):
                sample_lidar(pipe, noise, positive, negative, bank, rewards, 1.,
                             num_steps=3, diagnostics=diagnostics)
        finally:
            hook.remove()
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]['status'], 'error')
        self.assertEqual(diagnostics[0]['error_type'], 'FloatingPointError')
        for model in (pipe.unet, pipe.vae):
            parameter = next(model.parameters())
            parameter.requires_grad_(True)
            with self.assertRaises(ValueError):
                sample_lidar(pipe, noise, positive, negative, bank, rewards, 1., num_steps=3)
            parameter.requires_grad_(False)
            parameter.grad = torch.zeros_like(parameter)
            with self.assertRaises(ValueError):
                sample_lidar(pipe, noise, positive, negative, bank, rewards, 1., num_steps=3)
            parameter.grad = None


if __name__ == '__main__':
    unittest.main(verbosity=2)
