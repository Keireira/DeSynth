"""Exercise the installed Diffusers img2img path without model weights or a GPU."""
from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
from diffusers import QwenImageImg2ImgPipeline
from diffusers.image_processor import VaeImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Test the same standalone modules imported by the executable worker.
import runtime as ds  # pylint: disable=wrong-import-position


def configured_scheduler():
    """Get the production scheduler while replacing only expensive model loads."""
    pipe = Mock()
    captured = {}

    def load(_loader, **kwargs):
        if _loader is ds.hf_hub_download:
            return str(Path("cache/snapshots/revision") / kwargs["filename"])
        if "scheduler" in kwargs:
            captured["scheduler"] = kwargs["scheduler"]
            return pipe
        return Mock()

    with patch.object(ds, "apply_gguf_patches"), \
            patch.object(ds, "load_cached", side_effect=load), \
            patch.object(ds.AutoencoderKLQwenImage, "from_pretrained", return_value=Mock()), \
            patch.object(ds, "place_pipeline"), \
            patch("safetensors.torch.load_file", return_value={}):
        ds.build_pipeline(Path("test.gguf"), Path("test.safetensors"), "cpu", "none", "float32", .8)
    return captured["scheduler"]


class ZeroVelocityTransformer:
    """Keep the initially mixed latents intact so the exact noise is observable."""
    config = SimpleNamespace(in_channels=64, guidance_embeds=False)

    def __init__(self):
        self.calls = []
        self.contexts = []

    def cache_context(self, name):
        self.contexts.append(name)
        return nullcontext()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return (torch.zeros_like(kwargs["hidden_states"]),)


class PipelineProbe:  # pylint: disable=too-many-instance-attributes
    """Use real preprocessing, timestep selection, latent mixing and Euler steps."""
    check_inputs = QwenImageImg2ImgPipeline.check_inputs
    encode_prompt = QwenImageImg2ImgPipeline.encode_prompt
    get_timesteps = QwenImageImg2ImgPipeline.get_timesteps
    prepare_latents = QwenImageImg2ImgPipeline.prepare_latents
    # Exercise the pinned pipeline's actual private preprocessing interfaces.
    _pack_latents = staticmethod(QwenImageImg2ImgPipeline._pack_latents)  # pylint: disable=protected-access
    _callback_tensor_inputs = QwenImageImg2ImgPipeline._callback_tensor_inputs  # pylint: disable=protected-access
    attention_kwargs = property(lambda self: self._attention_kwargs)
    interrupt = property(lambda self: self._interrupt)

    def __init__(self):
        self._execution_device = "cpu"
        self._attention_kwargs = None
        self._interrupt = False
        self.vae_scale_factor = 8
        self.latent_channels = 16
        self.default_sample_size = 128
        self.image_processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=16)
        self.scheduler = configured_scheduler()
        self.transformer = ZeroVelocityTransformer()
        self.encoded_sizes = []
        self.call_params = []

    def __call__(self, **kwargs):
        self.call_params.append(kwargs)
        # Skip only VAE decoding; the returned latents expose source/noise mixing.
        return QwenImageImg2ImgPipeline.__call__(self, **kwargs, output_type="latent")

    # The real pipeline supplies generator and total as keyword arguments.
    def _encode_vae_image(self, image, generator):  # pylint: disable=unused-argument
        self.encoded_sizes.append(tuple(image.shape))
        return torch.full((1, 16, 1, image.shape[-2] // 8, image.shape[-1] // 8), 2.)

    def progress_bar(self, total):  # pylint: disable=unused-argument
        return nullcontext(SimpleNamespace(update=lambda: None))

    def maybe_free_model_hooks(self):
        pass


class PipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.pipe = PipelineProbe()
        self.embeds = {
            "prompt_embeds": torch.ones((2, 3)),
            "prompt_embeds_mask": torch.ones(2, dtype=torch.int64),
            "negative_prompt_embeds": torch.zeros((2, 3)),
            "negative_prompt_embeds_mask": torch.ones(2, dtype=torch.int64),
        }

    def sample(self, size, denoise, steps, seed, callback=None):
        return ds.sample(self.pipe, Image.new("RGB", size), self.embeds,
                          denoise=denoise, steps=steps, seed=seed,
                          callback_on_step_end=callback)

    def test_source_noise_dimensions_callbacks_and_cfg_through_real_pipeline(self):
        def record_step(events, _pipe, step, timestep, values):
            events.append((step, timestep.item()))
            return values

        # Reuse one scheduler across changing strengths, lengths, seeds and sizes.
        # This also catches cached begin/step indexes leaking between passes/jobs.
        cases = [(.25, 32), (.125, 32), (.01, 8), (.8, 16), (1., 8), (.25, 32)]
        for size in ((32, 16), (64, 64), (80, 112)):
            for index, (denoise, steps) in enumerate(cases):
                with self.subTest(size=size, denoise=denoise, steps=steps):
                    seed = 42 + index
                    callbacks = []

                    self.pipe.transformer.calls.clear()
                    self.pipe.transformer.contexts.clear()
                    with self.assertNoLogs("diffusers", level="WARNING"):
                        result = self.sample(size, denoise, steps, seed, partial(record_step, callbacks))
                    count = ds.effective_img2img_steps(steps, denoise)
                    noise = torch.randn((1, 1, 16, size[1] // 8, size[0] // 8),
                                        generator=torch.Generator().manual_seed(seed))
                    expected = self.pipe._pack_latents(  # pylint: disable=protected-access
                        2 * (1 - denoise) + denoise * noise, 1, 16,
                        size[1] // 8, size[0] // 8,
                    )[0]
                    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
                    self.assertEqual(self.pipe.encoded_sizes[-1], (1, 3, 1, size[1], size[0]))
                    self.assertEqual(self.pipe.scheduler.step_index, count)
                    self.assertEqual([step for step, _ in callbacks], list(range(count)))
                    np.testing.assert_allclose([t for _, t in callbacks],
                                               np.linspace(denoise * 1000, 0, count + 1)[:-1],
                                               atol=1e-4)
                    self.assertEqual(self.pipe.transformer.contexts, ["cond"] * count)
                    self.assertEqual(len(self.pipe.transformer.calls), count)
                    self.assertTrue(all(call["guidance"] is None for call in self.pipe.transformer.calls))
                    self.assertEqual(self.pipe.call_params[-1]["true_cfg_scale"], 1.)
        self.assertEqual(len(self.pipe.encoded_sizes), 18)

    def test_partial_call_does_not_leave_scheduler_at_the_wrong_step(self):
        def interrupt(_pipe, step, _timestep, values):
            if step == 1:
                raise RuntimeError("test interruption")
            return values

        with self.assertRaisesRegex(RuntimeError, "test interruption"):
            self.sample((32, 16), .25, 32, 42, interrupt)
        actual = self.sample((32, 16), .125, 32, 42)
        self.assertEqual(self.pipe.scheduler.step_index, 4)
        self.pipe = PipelineProbe()
        expected = self.sample((32, 16), .125, 32, 42)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
