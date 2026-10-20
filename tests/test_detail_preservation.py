"""Noise calibration and contour restoration without model weights or GPU work."""
from __future__ import annotations

import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np
from PIL import Image

import desynth as ds
from processing import finish_image


class NoiseCalibrationTests(unittest.TestCase):
    @staticmethod
    def scheduler():
        return ds.FlowMatchEulerDiscreteScheduler(
            base_shift=math.log(ds.LIGHTNING_SCHEDULER_SHIFT),
            max_shift=math.log(ds.LIGHTNING_SCHEDULER_SHIFT),
            shift_terminal=None, use_dynamic_shifting=True,
            time_shift_type="exponential",
        )

    def test_installed_scheduler_starts_at_requested_noise_and_keeps_update_count(self):
        for steps in (1, 8, 16, 32, 100):
            for denoise in (.01, .05, .1, .125, .25, .275, .5, .99, 1):
                with self.subTest(steps=steps, denoise=denoise):
                    raw = ds._img2img_sigmas(steps, denoise)
                    actual = ds.effective_img2img_steps(steps, denoise)
                    self.assertEqual(len(raw), actual)
                    self.assertTrue(np.all((raw > 0) & (raw <= 1)))
                    scheduler = self.scheduler()
                    scheduler.set_timesteps(actual, sigmas=raw,
                                            mu=math.log(ds.LIGHTNING_SCHEDULER_SHIFT), device="cpu")
                    expected = np.linspace(denoise, 0, actual + 1)
                    np.testing.assert_allclose(scheduler.sigmas.numpy(), expected, atol=2e-7)
                    timesteps, count = ds.QwenImageImg2ImgPipeline.get_timesteps(
                        SimpleNamespace(scheduler=scheduler), actual, 1.0, "cpu",
                    )
                    self.assertEqual(count, actual)
                    self.assertEqual(len(timesteps), actual)
                    mixed = scheduler.scale_noise(ds.torch.zeros((1, 1)), timesteps[:1],
                                                  noise=ds.torch.ones((1, 1)))
                    self.assertAlmostEqual(mixed.item(), denoise, places=6)

    def test_quarter_denoise_is_not_doubled_by_lightning_shift(self):
        scheduler = self.scheduler()
        scheduler.set_timesteps(32, sigmas=np.linspace(1, 1 / 32, 32),
                                mu=math.log(ds.LIGHTNING_SCHEDULER_SHIFT), device="cpu")
        self.assertAlmostEqual(scheduler.sigmas[24].item(), .5)
        raw = ds._img2img_sigmas(32, .25)
        scheduler.set_timesteps(8, sigmas=raw,
                                mu=math.log(ds.LIGHTNING_SCHEDULER_SHIFT), device="cpu")
        self.assertAlmostEqual(scheduler.sigmas[0].item(), .25)
        self.assertEqual(len(scheduler.sigmas), 9)

    def test_invalid_schedule_fails_before_pipeline_work(self):
        for steps, denoise in ((0, .25), (-1, .25), (2.5, .25), (True, .25),
                               (32, 0), (32, -.1), (32, 1.01), (32, True),
                               (32, float("nan")), (32, float("inf")), (1, 1e-30)):
            with self.subTest(steps=steps, denoise=denoise), self.assertRaises(ValueError):
                ds._img2img_sigmas(steps, denoise)

    def test_sampling_passes_keep_seed_embeddings_size_and_existing_adapter(self):
        image = Image.new("RGB", (32, 16), (60, 70, 80))
        embeds = {
            "prompt_embeds": ds.torch.ones((2, 3)),
            "negative_prompt_embeds": ds.torch.zeros((2, 3)),
            "prompt_embeds_mask": ds.torch.ones(2, dtype=ds.torch.int64),
            "negative_prompt_embeds_mask": ds.torch.ones(2, dtype=ds.torch.int64),
            "meta": {"not": "a tensor"},
        }
        pipe = Mock(side_effect=lambda **kwargs: SimpleNamespace(images=[kwargs["image"]]))
        result = ds.run_once(pipe, image, embeds, denoise=.25, steps=32, seed=42, passes=3)
        self.assertIs(result, image)
        self.assertEqual(pipe.call_count, 3)
        self.assertEqual([call.kwargs["generator"].initial_seed() for call in pipe.call_args_list],
                         [42, 43, 44])
        for call in pipe.call_args_list:
            args = call.kwargs
            self.assertEqual(args["num_inference_steps"], 8)
            self.assertEqual(args["strength"], 1)
            self.assertEqual((args["width"], args["height"]), image.size)
            self.assertEqual(args["true_cfg_scale"], ds.CFG)
            np.testing.assert_allclose(args["sigmas"], ds._img2img_sigmas(32, .25))
            self.assertTrue(ds.torch.equal(args["prompt_embeds"], embeds["prompt_embeds"].unsqueeze(0)))
            self.assertTrue(ds.torch.equal(args["prompt_embeds_mask"], embeds["prompt_embeds_mask"].unsqueeze(0)))
        pipe.set_adapters.assert_not_called()


class ContourRestorationTests(unittest.TestCase):
    @staticmethod
    def contours():
        x = np.arange(192)[None, :]
        plane = np.broadcast_to(np.where((x // 24) % 2, 195, 40), (128, 192))
        original = np.repeat(plane[..., None], 3, axis=-1).astype(np.uint8)
        clean = cv2.GaussianBlur(original, (0, 0), sigmaX=3)
        return clean, original

    def test_edge_mode_returns_more_source_contour_than_gaussian(self):
        clean, original = self.contours()
        for sigma in (.5, 1, 1.95, 3, 6):
            with self.subTest(sigma=sigma):
                gaussian = ds._restore_gaussian(clean, original, sigma)
                edge = ds._restore_edge(clean, original, sigma)
                gaussian_error = np.abs(gaussian - original).mean()
                edge_error = np.abs(edge - original).mean()
                self.assertLess(edge_error, gaussian_error)
                if sigma == 1.95:
                    self.assertLess(edge_error, gaussian_error * .8)
                self.assertGreater(edge_error, 0)  # Restoration still includes generated content.

    def test_explicit_edge_sigma_cannot_restore_less_than_flat_region(self):
        clean, original = self.contours()
        actual = ds._restore_edge(clean, original, 1.95, sigma_edge=1)
        expected = ds._restore_gaussian(clean, original, 1.95)
        np.testing.assert_allclose(actual, expected, atol=2e-5)

    def test_edge_band_is_bounded_at_the_largest_allowed_sigma(self):
        clean, original = self.contours()
        gaussian = ds._restore_gaussian(clean, original, 10)
        edge = ds._restore_edge(clean, original, 10)
        np.testing.assert_allclose(edge, gaussian, atol=2e-5)

    def test_gaussian_formula_is_unchanged(self):
        clean, original = self.contours()
        sigma = 1.95
        source = original.astype(np.float32)
        expected = (cv2.GaussianBlur(clean.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
                    + source - cv2.GaussianBlur(source, (0, 0), sigmaX=sigma, sigmaY=sigma)).clip(0, 255)
        np.testing.assert_allclose(ds._restore_gaussian(clean, original, sigma), expected, atol=2e-5)

    def test_both_modes_keep_dimensions_and_every_source_alpha_value(self):
        clean, original = self.contours()
        source = Image.fromarray(original).resize((197, 131)).convert("RGBA")
        alpha = np.arange(source.width * source.height, dtype=np.uint32).astype(np.uint8)
        source.putalpha(Image.frombytes("L", source.size, alpha.tobytes()))
        for mode in ("gaussian", "edge"):
            with self.subTest(mode=mode):
                params = dict(restore=True, restore_sigma=1.95, restore_mode=mode, unsharp=0)
                final = finish_image(Image.fromarray(clean), source, params, ds._restore)
                self.assertEqual(final.mode, "RGBA")
                self.assertEqual(final.size, source.size)
                self.assertEqual(final.getchannel("A").tobytes(), source.getchannel("A").tobytes())


if __name__ == "__main__":
    unittest.main()
