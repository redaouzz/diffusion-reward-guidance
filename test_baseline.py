"""Tests CPU sans téléchargement : python -m unittest test_baseline -v.

Les petits réseaux aléatoires vérifient le câblage du sampler, pas la qualité
de SD1.5 ni sa faisabilité sur T4. Ces deux points restent à tester dans Colab.
"""

import unittest
from types import SimpleNamespace

import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel

from baseline import decode_latents, ddim_step, make_initial_noise, sample_ddim


def make_test_scheduler(**changes):
    """Configuration de test SD1.5 ; le notebook lit celle du checkpoint."""
    config = dict(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
        prediction_type="epsilon",
        timestep_spacing="leading",
    )
    config.update(changes)
    return DDIMScheduler(**config)


class DDIMStepTests(unittest.TestCase):
    def test_all_50_steps_match_diffusers_float32_and_float16(self):
        """Même arithmétique et mêmes coefficients que le pas officiel."""
        for dtype in (torch.float32, torch.float16):
            scheduler = make_test_scheduler()
            scheduler.set_timesteps(50)
            generator = torch.Generator(device="cpu").manual_seed(12)
            for timestep in scheduler.timesteps:
                with self.subTest(dtype=dtype, timestep=int(timestep)):
                    latents = torch.randn((1, 4, 8, 8), generator=generator, dtype=dtype)
                    noise = torch.randn((1, 4, 8, 8), generator=generator, dtype=dtype)
                    actual, predicted_x0 = ddim_step(latents, noise, int(timestep), scheduler)
                    expected = scheduler.step(noise, int(timestep), latents, eta=0.0)
                    # Même ordre d'opérations : aucune marge fp16 ne doit être
                    # nécessaire pour un pas isolé. Le test GPU est distinct.
                    torch.testing.assert_close(actual, expected.prev_sample, rtol=0, atol=0)
                    torch.testing.assert_close(
                        predicted_x0, expected.pred_original_sample, rtol=0, atol=0
                    )
                    self.assertEqual(actual.shape, latents.shape)
                    self.assertTrue(torch.isfinite(actual).all())

    def test_last_step_and_clipping_follow_configuration(self):
        for set_alpha_to_one in (False, True):
            for clip_sample in (False, True):
                with self.subTest(set_alpha_to_one=set_alpha_to_one, clip_sample=clip_sample):
                    scheduler = make_test_scheduler(
                        set_alpha_to_one=set_alpha_to_one, clip_sample=clip_sample
                    )
                    scheduler.set_timesteps(50)
                    timestep = int(scheduler.timesteps[-1])
                    latents = torch.linspace(-3.0, 3.0, 64).reshape(1, 4, 4, 4)
                    noise = torch.full_like(latents, 0.25)
                    actual, predicted_x0 = ddim_step(latents, noise, timestep, scheduler)
                    expected = scheduler.step(noise, timestep, latents, eta=0.0)
                    torch.testing.assert_close(actual, expected.prev_sample, rtol=0, atol=0)
                    torch.testing.assert_close(
                        predicted_x0, expected.pred_original_sample, rtol=0, atol=0
                    )
                    if clip_sample:
                        self.assertLessEqual(predicted_x0.abs().max().item(), 1.0)

    def test_unsupported_prediction_and_thresholding_are_rejected(self):
        for config in ({"prediction_type": "v_prediction"}, {"thresholding": True}):
            with self.subTest(config=config):
                scheduler = make_test_scheduler(**config)
                scheduler.set_timesteps(50)
                latents = torch.zeros(1, 4, 4, 4)
                with self.assertRaises((ValueError, NotImplementedError)):
                    ddim_step(latents, latents, int(scheduler.timesteps[0]), scheduler)

    def test_initial_noise_shape_finiteness_and_seed(self):
        # Seuls les attributs nécessaires au tirage du bruit sont simulés.
        pipe = SimpleNamespace(
            _execution_device=torch.device("cpu"),
            unet=SimpleNamespace(dtype=torch.float32, config=SimpleNamespace(in_channels=4)),
            vae_scale_factor=8,
        )
        first = make_initial_noise(pipe, seed=123)
        repeated = make_initial_noise(pipe, seed=123)
        other_seed = make_initial_noise(pipe, seed=124)
        self.assertEqual(first.shape, (1, 4, 64, 64))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, other_seed))


class TinyPipelineTests(unittest.TestCase):
    def test_sampler_and_decode_match_real_diffusers_pipeline(self):
        """Vrais composants aléatoires, 3 pas CPU, aucune donnée préentraînée."""
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(2, previous_threads))
        try:
            torch.manual_seed(17)
            unet = UNet2DConditionModel(
                sample_size=8,
                in_channels=4,
                out_channels=4,
                layers_per_block=1,
                block_out_channels=(8, 16),
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                cross_attention_dim=8,
                attention_head_dim=2,
                norm_num_groups=4,
            )
            vae = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                down_block_types=("DownEncoderBlock2D",),
                up_block_types=("UpDecoderBlock2D",),
                block_out_channels=(8,),
                latent_channels=4,
                norm_num_groups=4,
                sample_size=8,
                scaling_factor=0.18215,
            )
            pipe = StableDiffusionPipeline(
                vae=vae,
                text_encoder=None,
                tokenizer=None,
                unet=unet,
                scheduler=make_test_scheduler(),
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            ).to("cpu")
            pipe.unet.eval().requires_grad_(False)
            pipe.vae.eval().requires_grad_(False)
            pipe.set_progress_bar_config(disable=True)
            prompt_embeds = torch.randn(1, 4, 8)
            negative_prompt_embeds = torch.randn(1, 4, 8)
            initial_noise = make_initial_noise(pipe, seed=31, height=8, width=8)
            initial_copy = initial_noise.clone()

            official_arguments = dict(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                latents=initial_noise.clone(),
                height=8,
                width=8,
                num_inference_steps=3,
                guidance_scale=7.5,
                guidance_rescale=0.0,
                eta=0.0,
            )
            expected_latents = pipe(**official_arguments, output_type="latent").images
            actual_latents, _ = sample_ddim(
                pipe, initial_noise, prompt_embeds, negative_prompt_embeds,
                num_steps=3, guidance_scale=7.5,
            )
            repeated_latents, _ = sample_ddim(
                pipe, initial_noise, prompt_embeds, negative_prompt_embeds,
                num_steps=3, guidance_scale=7.5,
            )
            torch.testing.assert_close(actual_latents, expected_latents, rtol=1e-6, atol=1e-6)
            self.assertTrue(torch.equal(actual_latents, repeated_latents))
            self.assertTrue(torch.equal(initial_noise, initial_copy))
            self.assertTrue(torch.isfinite(actual_latents).all())
            self.assertFalse(actual_latents.requires_grad)

            expected_image = pipe(**official_arguments, output_type="pt").images
            actual_image = decode_latents(pipe, actual_latents)
            torch.testing.assert_close(actual_image, expected_image, rtol=1e-6, atol=1e-6)
            self.assertEqual(actual_image.shape, (1, 3, 8, 8))
            self.assertTrue(torch.isfinite(actual_image).all())
            self.assertGreaterEqual(actual_image.min().item(), 0.0)
            self.assertLessEqual(actual_image.max().item(), 1.0)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main(verbosity=2)
