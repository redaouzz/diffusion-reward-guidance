"""Tests CPU : python -m unittest test_rewards -v.

Aucun poids téléchargé. Le petit VAE aléatoire teste autograd, pas SD1.5.
Le modèle ImageReward et le chemin complet sur T4 se vérifient dans le notebook.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from diffusers import AutoencoderKL
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from baseline import decode_latents
from rewards import (
    check_latent_gradient,
    decode_latents_differentiable,
    luminance_map,
    luminance_reward,
    make_disk_mask,
    preprocess_imagereward,
    spatial_mask_reward,
    symmetry_reward,
)


def make_test_vae():
    """Petit décodeur réel, aléatoire et gelé ; aucun poids SD1.5."""
    torch.manual_seed(18)
    return AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(8,), latent_channels=4,
        norm_num_groups=4, sample_size=8, scaling_factor=0.18215,
    ).eval().requires_grad_(False)


class AnalyticRewardTests(unittest.TestCase):
    def test_values_coefficients_and_maxima(self):
        images = torch.zeros(5, 3, 4, 6)
        images[1] = 1
        images[2, 0] = 1
        images[3, 1] = 1
        images[4, 2] = 1
        expected = torch.tensor([0.0, 1.0, 0.299, 0.587, 0.114])
        torch.testing.assert_close(luminance_map(images).mean((1, 2)), expected)
        torch.testing.assert_close(
            luminance_reward(images[:2], target=0.7), torch.tensor([-0.49, -0.09])
        )
        target_images = torch.full((2, 3, 4, 6), 0.7)
        torch.testing.assert_close(luminance_reward(target_images), torch.zeros(2), atol=1e-12, rtol=0)

        asymmetric = torch.zeros(2, 3, 4, 6)
        asymmetric[:, :, :, 3:] = 1
        torch.testing.assert_close(symmetry_reward(asymmetric), -torch.ones(2))
        symmetric = (asymmetric + asymmetric.flip(-1)) / 2
        torch.testing.assert_close(symmetry_reward(symmetric), torch.zeros(2), atol=0, rtol=0)

    def test_disk_and_mask_normalization(self):
        mask = make_disk_mask(9, 11)
        self.assertEqual(mask.shape, (9, 11))
        self.assertEqual(mask[4, 5].item(), 1.0)
        self.assertEqual(mask[0, 0].item(), 0.0)
        self.assertTrue(torch.equal(mask, mask.flip(-1)))
        self.assertTrue(torch.equal(mask, mask.flip(-2)))
        matched_images = mask[None, None].repeat(2, 3, 1, 1)
        torch.testing.assert_close(
            spatial_mask_reward(matched_images, mask), torch.zeros(2), atol=1e-12, rtol=0
        )
        # Une image noire contre un masque uniformément blanc a une MSE de 1.
        torch.testing.assert_close(
            spatial_mask_reward(torch.zeros(2, 3, 9, 11), torch.ones(9, 11)),
            -torch.ones(2),
        )

    def test_gradients_match_closed_form_in_float32(self):
        height, width = 5, 7
        generator = torch.Generator().manual_seed(10)
        images = (0.2 + 0.6 * torch.rand(2, 3, height, width, generator=generator)).requires_grad_(True)
        weights = images.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        luminance = (images.detach() * weights).sum(1)
        mask = make_disk_mask(height, width)

        # La somme des rewards du batch n'introduit aucune division par B.
        luminance_gradient = (
            -2 * (luminance.mean((1, 2)) - 0.7)[:, None, None, None]
            * weights / (height * width)
        ).expand_as(images)
        # flip dépend aussi de l'image : les deux contributions donnent 4.
        symmetry_gradient = -4 * (images.detach() - images.detach().flip(-1)) / (3 * height * width)
        mask_gradient = -2 * (luminance - mask)[:, None] * weights / (height * width)

        cases = (
            ("luminance", luminance_reward(images), luminance_gradient),
            ("symmetry", symmetry_reward(images), symmetry_gradient),
            ("mask", spatial_mask_reward(images, mask), mask_gradient),
        )
        for name, reward, expected in cases:
            with self.subTest(reward=name):
                self.assertEqual(reward.shape, (2,))
                actual = torch.autograd.grad(reward.sum(), images)[0]
                self.assertTrue(torch.isfinite(actual).all())
                self.assertGreater(actual.norm().item(), 0)
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_directional_derivatives_in_float32(self):
        generator = torch.Generator().manual_seed(23)
        images = (0.2 + 0.6 * torch.rand(1, 3, 4, 5, generator=generator)).requires_grad_(True)
        mask = make_disk_mask(4, 5)
        rewards = (luminance_reward, symmetry_reward, lambda x: spatial_mask_reward(x, mask))
        for reward_fn in rewards:
            gradient = torch.autograd.grad(reward_fn(images).sum(), images)[0]
            direction = gradient / gradient.norm()
            epsilon = 1e-3
            with torch.no_grad():
                finite_difference = (
                    reward_fn(images + epsilon * direction).sum()
                    - reward_fn(images - epsilon * direction).sum()
                ) / (2 * epsilon)
            derivative = (gradient * direction).sum()
            # Différence de deux valeurs fp32 / 2e-3 : tolérance plus large
            # que le test analytique, sans différence finie en fp16.
            torch.testing.assert_close(finite_difference, derivative, rtol=2e-3, atol=1e-5)
            self.assertGreater(finite_difference.item(), 0)

    def test_shapes_and_nonfinite_inputs_are_rejected(self):
        mask = torch.zeros(4, 5)
        rewards = (
            luminance_map, luminance_reward, symmetry_reward,
            lambda x: spatial_mask_reward(x, mask),
        )
        for bad_shape in ((3, 4, 5), (1, 1, 4, 5)):
            for reward_fn in rewards:
                with self.assertRaises(ValueError):
                    reward_fn(torch.zeros(bad_shape))
        for invalid in (float("nan"), float("inf")):
            images = torch.zeros(1, 3, 4, 5)
            images[0, 0, 0, 0] = invalid
            for reward_fn in rewards:
                with self.assertRaises((ValueError, FloatingPointError)):
                    reward_fn(images)
        with self.assertRaises(ValueError):
            spatial_mask_reward(torch.zeros(1, 3, 4, 5), torch.zeros(5, 4))

    def test_reductions_promote_half_and_preserve_double(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                images = torch.linspace(0.1, 0.9, 120, dtype=dtype).reshape(2, 3, 4, 5).requires_grad_(True)
                mask = make_disk_mask(4, 5, dtype=dtype)
                expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
                self.assertEqual(luminance_map(images).dtype, expected_dtype)
                rewards = (
                    luminance_reward(images), symmetry_reward(images), spatial_mask_reward(images, mask)
                )
                for reward in rewards:
                    self.assertEqual(reward.dtype, expected_dtype)
                    self.assertEqual(reward.shape, (2,))
                    self.assertTrue(torch.isfinite(reward).all())
                gradient = torch.autograd.grad(sum(reward.sum() for reward in rewards), images)[0]
                self.assertTrue(torch.isfinite(gradient).all())


class DifferentiablePathTests(unittest.TestCase):
    def test_imagereward_preprocessing_geometry_and_gradient(self):
        # Référence torchvision tensorielle : ce test ne charge pas ImageReward.
        images = torch.linspace(0.1, 0.9, 2 * 3 * 240 * 320).reshape(2, 3, 240, 320).requires_grad_(True)
        reference = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        actual = preprocess_imagereward(images)
        torch.testing.assert_close(actual, reference(images), rtol=0, atol=0)
        self.assertEqual(actual.shape, (2, 3, 224, 224))
        gradient = torch.autograd.grad(actual.square().mean(), images)[0]
        self.assertEqual(gradient.shape, images.shape)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.norm().item(), 0)

    def test_frozen_vae_forward_and_latent_gradients(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(2, previous_threads))
        try:
            vae = make_test_vae()
            latents = (0.1 * torch.randn(1, 4, 8, 8)).requires_grad_(True)
            actual_image = decode_latents_differentiable(vae, latents)
            reference_image = decode_latents(SimpleNamespace(vae=vae), latents)
            torch.testing.assert_close(actual_image, reference_image, rtol=0, atol=0)
            self.assertTrue(actual_image.requires_grad)
            self.assertFalse(reference_image.requires_grad)
            self.assertGreaterEqual(actual_image.min().item(), 0)
            self.assertLessEqual(actual_image.max().item(), 1)

            mask = make_disk_mask(8, 8)
            rewards = (luminance_reward, symmetry_reward, lambda x: spatial_mask_reward(x, mask))
            for reward_fn in rewards:
                clean_latent = latents.detach().clone().requires_grad_(True)
                image = decode_latents_differentiable(vae, clean_latent)
                reward = reward_fn(image).sum()
                gradient = torch.autograd.grad(reward, clean_latent)[0]
                self.assertEqual(gradient.shape, clean_latent.shape)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.norm().item(), 0)
                direction = gradient / gradient.norm()
                with torch.no_grad():
                    moved_image = decode_latents_differentiable(vae, clean_latent + 1e-3 * direction)
                    improved_reward = reward_fn(moved_image).sum()
                self.assertGreater(improved_reward.item(), reward.item())
                self.assertTrue(all(not parameter.requires_grad for parameter in vae.parameters()))
                self.assertTrue(all(parameter.grad is None for parameter in vae.parameters()))
        finally:
            torch.set_num_threads(previous_threads)

    def test_vae_checkpointing_preserves_forward_and_latent_gradient(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(2, previous_threads))
        try:
            vae = make_test_vae()
            latents = (0.1 * torch.randn(1, 4, 8, 8)).requires_grad_(True)
            reference_image = decode_latents_differentiable(vae, latents)
            reference_gradient = torch.autograd.grad(luminance_reward(reference_image).sum(), latents)[0]

            # Le notebook utilise le checkpointing pour échanger du calcul
            # contre des activations conservées ; il doit préserver la dérivée.
            vae.enable_gradient_checkpointing()
            checkpoint_latents = latents.detach().clone().requires_grad_(True)
            checkpoint_image = decode_latents_differentiable(vae, checkpoint_latents)
            checkpoint_gradient = torch.autograd.grad(
                luminance_reward(checkpoint_image).sum(), checkpoint_latents,
            )[0]
            torch.testing.assert_close(checkpoint_image, reference_image, rtol=0, atol=0)
            torch.testing.assert_close(checkpoint_gradient, reference_gradient, rtol=1e-6, atol=1e-7)
            self.assertTrue(torch.isfinite(checkpoint_gradient).all())
            self.assertGreater(checkpoint_gradient.norm().item(), 0)
            self.assertTrue(all(parameter.grad is None for parameter in vae.parameters()))
        finally:
            torch.set_num_threads(previous_threads)

    def test_latent_gradient_diagnostic_reports_ascent_without_parameter_grads(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(2, previous_threads))
        try:
            vae = make_test_vae()
            latents = 0.1 * torch.randn(1, 4, 8, 8)
            initial_latents = latents.clone()
            # Le diagnostic ouvre lui-même son graphe, même dans une cellule
            # de référence sous no_grad. Il ne modifie pas le latent fourni.
            with torch.no_grad():
                report, gradient = check_latent_gradient(
                    vae, latents, luminance_reward, min_consecutive_steps=2,
                )
            self.assertTrue(report["passed"], report)
            self.assertEqual(report["gradient_shape"], list(latents.shape))
            self.assertTrue(report["gradient_finite"])
            self.assertGreater(report["gradient_l2"], 0)
            self.assertTrue(torch.equal(latents, initial_latents))
            self.assertFalse(gradient.requires_grad)
            self.assertEqual(gradient.shape, latents.shape)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertEqual(report["gradient_l2"], gradient.norm().item())
            # Plusieurs pas évitent de dépendre d'une seule différence fp32.
            self.assertEqual(len(report["directional_checks"]), 3)
            self.assertEqual(report["min_consecutive_steps"], 2)
            self.assertGreaterEqual(report["max_consecutive_passing_steps"], 2)
            self.assertTrue(report["vae_parameters_frozen"])
            self.assertTrue(report["vae_parameter_grads_none"])
            self.assertTrue(all(parameter.grad is None for parameter in vae.parameters()))

            # Un gradient nul ne doit pas être déclaré comme une ascension validée.
            stationary_report, stationary_gradient = check_latent_gradient(
                vae, latents, lambda image: 0 * luminance_reward(image),
            )
            self.assertFalse(stationary_report["passed"])
            self.assertEqual(stationary_report["directional_checks"], [])
            self.assertEqual(stationary_gradient.norm().item(), 0)
        finally:
            torch.set_num_threads(previous_threads)

    def test_latent_gradient_rejects_invalid_step_grids(self):
        vae = make_test_vae()
        latents = torch.zeros(1, 4, 8, 8)
        invalid_cases = (
            ((0.01,), 2),              # Impossible d'obtenir deux pas voisins.
            ((0.01, 0.01), 2),         # Répéter le même pas ne compte pas deux fois.
            ((0.001, 0.01), 2),        # La grille doit être strictement décroissante.
            ((0.01, 0.0), 1),
            ((0.01, float("nan")), 1),
        )
        for steps, required in invalid_cases:
            with self.subTest(steps=steps, required=required):
                with self.assertRaises(ValueError):
                    check_latent_gradient(
                        vae, latents, luminance_reward,
                        step_sizes=steps, min_consecutive_steps=required,
                    )

    def test_directional_successes_must_be_consecutive_and_include_ascent(self):
        # Test logique du diagnostic seulement : remplacer le décodage par
        # l'identité permet des fonctions polynomiales dont les résultats sont connus.
        vae = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        latents = torch.zeros(1, 1, 1, 1)

        def identity_decode(_vae, point):
            self.assertTrue(torch.is_grad_enabled())
            self.assertTrue(point.requires_grad and point.is_leaf)
            return point

        with patch("rewards.decode_latents_differentiable", side_effect=identity_decode):
            for failure in ("derivative", "ascent"):
                with self.subTest(failure=failure):
                    def polynomial_reward(image):
                        x = image[:, 0, 0, 0]
                        polynomial = x.square() * (x.square() - 9) * (x.square() - 1)
                        factor = x if failure == "derivative" else 4
                        return x - polynomial * factor / 60

                    # R'(0)=1 dans les deux cas. À h=3 et h=1, R(±h)=±h.
                    # À h=2 : dérivée centrale=2, ou R(+h)=6 et R(-h)=2.
                    report, _ = check_latent_gradient(
                        vae, latents, polynomial_reward,
                        step_sizes=(3.0, 2.0, 1.0), min_consecutive_steps=2,
                    )
                    successes = [
                        check["derivative_matches"] and check["local_ascent"]
                        for check in report["directional_checks"]
                    ]
                    self.assertEqual(successes, [True, False, True])
                    self.assertEqual(report["max_consecutive_passing_steps"], 1)
                    self.assertFalse(report["passed"])

            # Deux pas réellement voisins passent pour la fonction linéaire.
            report, _ = check_latent_gradient(
                vae, latents, lambda image: image[:, 0, 0, 0],
                step_sizes=(2.0, 1.0), min_consecutive_steps=2,
            )
            self.assertEqual(report["max_consecutive_passing_steps"], 2)
            self.assertTrue(report["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
