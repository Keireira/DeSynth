"""Inference-boundary regressions without model weights, GPU work, or network."""
from __future__ import annotations

from contextlib import nullcontext, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image

WORKER = Path(__file__).resolve().parent.parent / "worker.py"
sys.path.insert(0, str(WORKER.parent))
import worker as inference
import runtime
from diffusers_compat import apply_gguf_patches, load_cached


class ProtocolTests(unittest.TestCase):
    def test_only_actual_memory_errors_trigger_smaller_model_fallback(self):
        for error in (MemoryError(), RuntimeError("CUDA out of memory"), RuntimeError("OOM"),
                      RuntimeError("memory allocation failed")):
            self.assertTrue(inference.is_oom(error))
        for error in (RuntimeError("boom: invalid configuration"), ValueError("out of memory")):
            self.assertFalse(inference.is_oom(error))

    def test_invalid_commands_do_not_break_following_commands(self):
        commands = '\n'.join(("not json", "[]", '{"id":1,"op":"unknown"}', '{"id":2,"op":"release"}')) + '\n'
        result = subprocess.run([sys.executable, "-u", str(WORKER)], input=commands, text=True,
                                capture_output=True, timeout=15, check=True)
        frames = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([frame["event"] for frame in frames], ["error", "error", "error", "result"])
        self.assertEqual(frames[-1], {"id": 2, "event": "result", "result": {"released": True}})
        self.assertIn("JSONDecodeError", result.stderr)

    def test_probe_reports_missing_dependencies_and_keeps_cpu_capability(self):
        with patch.object(importlib.util, "find_spec", return_value=None):
            result = inference.probe()
        self.assertFalse(result["ready"])
        self.assertIn("torch", result["missing"])
        self.assertTrue(result["runtime"]["ready"])
        self.assertEqual(result["runtime"]["default_device"], "cpu")
        self.assertEqual(result["runtime"]["devices"][0]["precisions"], ["float32"])

    def test_validation_happens_before_loading_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ("model", "adapter", "embeds", "source"):
                (directory / name).write_bytes(b"placeholder")
            request = dict(transformer=str(directory / "model"), lightning=str(directory / "adapter"),
                           embeds=str(directory / "embeds"), source=str(directory / "source"),
                           output=str(directory / "output.png"), denoise=.25, steps=32, passes=1,
                           seed=42, device="cpu", offload="none", precision="float32", lightning_scale=.8)
            inference.validate_sample(request)
            for field, value in (("steps", True), ("passes", 0), ("seed", 2**63),
                                 ("denoise", float("nan")), ("lightning_scale", 0),
                                 ("precision", "float16"), ("source", "relative.png")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    inference.validate_sample({**request, field: value})


class SamplingTests(unittest.TestCase):
    @staticmethod
    def fixture(directory):
        transformer = directory / "model.gguf"
        transformer.write_bytes(b"model")
        lightning = directory / "lightning.safetensors"
        lightning.write_bytes(b"adapter")
        embeds = directory / "embeds.pt"
        torch.save({"prompt_embeds": torch.ones((2, 3)), "negative_prompt_embeds": torch.zeros((2, 3)),
                    "prompt_embeds_mask": torch.ones(2, dtype=torch.int64),
                    "negative_prompt_embeds_mask": torch.ones(2, dtype=torch.int64), "meta": {"ignored": True}}, embeds)
        source = directory / "source.png"
        Image.new("RGB", (32, 16), (30, 50, 70)).save(source)
        return dict(transformer=str(transformer), lightning=str(lightning), embeds=str(embeds), source=str(source),
                    output=str(directory / "clean.png"), denoise=.25, steps=32, passes=2, seed=42,
                    device="cpu", offload="none", precision="float32", lightning_scale=.8)

    def test_schedule_keeps_requested_noise_after_lightning_shift(self):
        for steps, denoise in ((32, .25), (32, .125), (8, .275), (1, 1)):
            with self.subTest(steps=steps, denoise=denoise):
                values = runtime.img2img_sigmas(steps, denoise)
                shifted = 3 * values / (1 + 2 * values)
                expected = np.linspace(denoise, 0, runtime.effective_img2img_steps(steps, denoise) + 1)[:-1]
                np.testing.assert_allclose(shifted, expected, atol=1e-12)
        with self.assertRaises(ValueError):
            runtime.img2img_sigmas(1, 1e-30)

    def test_sampling_preserves_embeds_seed_dimensions_and_physical_sigma(self):
        image = Image.new("RGB", (32, 16))
        embeds = {"prompt_embeds": torch.ones((2, 3)), "negative_prompt_embeds": torch.zeros((2, 3)),
                  "prompt_embeds_mask": torch.ones(2, dtype=torch.int64),
                  "negative_prompt_embeds_mask": torch.ones(2, dtype=torch.int64)}
        pipe = Mock(return_value=SimpleNamespace(images=[image]))
        callback = Mock()
        actual = runtime.sample(pipe, image, embeds, denoise=.25, steps=32, seed=42, callback_on_step_end=callback)
        self.assertIs(actual, image)
        args = pipe.call_args.kwargs
        self.assertEqual(args["generator"].initial_seed(), 42)
        self.assertEqual((args["width"], args["height"]), (32, 16))
        self.assertEqual(args["num_inference_steps"], 8)
        self.assertEqual(args["strength"], 1.0)
        self.assertIs(args["callback_on_step_end"], callback)
        self.assertTrue(torch.equal(args["prompt_embeds_mask"], embeds["prompt_embeds_mask"].unsqueeze(0)))
        np.testing.assert_allclose(args["sigmas"], runtime.img2img_sigmas(32, .25))

    def test_worker_caches_pipeline_by_weights_runtime_adapter_and_scale(self):
        with tempfile.TemporaryDirectory() as temporary:
            request = self.fixture(Path(temporary))
            pipe = Mock(num_timesteps=8)
            progress = Mock()
            samples = []

            def sample(selected, image, embeds, **kwargs):
                self.assertIs(selected, pipe)
                self.assertEqual(embeds["prompt_embeds_mask"].dtype, torch.int64)
                samples.append(kwargs["seed"])
                kwargs["callback_on_step_end"](pipe, 0, None, {})
                return image

            owner = runtime.Runtime()
            with patch.object(runtime, "build_pipeline", return_value=pipe) as build, patch.object(runtime, "sample", side_effect=sample):
                result = owner.run(request, progress)
                owner.run(request, progress)
                self.assertEqual(build.call_count, 1)
                self.assertEqual(samples, [42, 43, 42, 43])
                self.assertEqual(result["actual_steps"], 8)
                self.assertTrue(Path(request["output"]).is_file())
                with Image.open(request["output"]) as image:
                    self.assertEqual(image.size, (32, 16))
                    self.assertEqual(image.getpixel((0, 0)), (30, 50, 70))
                owner.run({**request, "lightning_scale": .7}, progress)
                self.assertEqual(build.call_count, 1)
                pipe.set_adapters.assert_called_once_with(["lightning"], adapter_weights=[.7])
                Path(request["lightning"]).write_bytes(b"new adapter bytes")
                owner.run(request, progress)
                self.assertEqual(build.call_count, 2)
                owner.run({**request, "precision": "bfloat16"}, progress)
                self.assertEqual(build.call_count, 3)
            self.assertTrue(any("step 1/8" in call.args[1] for call in progress.call_args_list))
            owner.release()
            self.assertIsNone(owner.pipe)
            self.assertIsNone(owner.embeds)


class CompatibilityTests(unittest.TestCase):
    def test_release_covers_each_cuda_device_after_runtime_switch(self):
        with (patch.object(torch.cuda, "is_available", return_value=True),
              patch.object(torch.cuda, "device_count", return_value=2),
              patch.object(torch.cuda, "device", side_effect=lambda _: nullcontext()) as device,
              patch.object(torch.cuda, "empty_cache") as empty,
              patch.object(torch.cuda, "ipc_collect") as collect,
              patch.object(torch.backends.mps, "is_available", return_value=False)):
            runtime.Runtime().release()
        self.assertEqual([call.args[0] for call in device.call_args_list], [0, 1])
        self.assertEqual(empty.call_count, 2)
        self.assertEqual(collect.call_count, 2)

    def test_lora_oom_is_not_retried_with_a_different_loader(self):
        pipe = Mock()
        pipe.load_lora_weights.side_effect = RuntimeError("out of memory")
        with (patch.object(runtime.QwenImageTransformer2DModel, "from_single_file", return_value=Mock()),
              patch.object(runtime.AutoencoderKLQwenImage, "from_pretrained", return_value=Mock()),
              patch.object(runtime.QwenImageImg2ImgPipeline, "from_pretrained", return_value=pipe),
              patch("safetensors.torch.load_file", return_value={"weight": object()}) as load_weights):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                runtime.build_pipeline(Path("model.gguf"), Path("adapter.safetensors"), "cpu", "none", "float32", .8)
            load_weights.assert_called_once_with("adapter.safetensors", device="cpu")
            pipe.load_lora_weights.assert_called_once()

    def test_cache_miss_retries_once_but_model_loading_failures_do_not(self):
        from huggingface_hub.errors import LocalEntryNotFoundError

        wrapper = OSError("Diffusers wrapper")
        wrapper.__cause__ = LocalEntryNotFoundError("Not cached")
        loader = Mock(side_effect=[wrapper, "loaded"])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(load_cached(loader, config="local"), "loaded")
        self.assertEqual([call.kwargs["local_files_only"] for call in loader.call_args_list], [True, False])
        for error in (MemoryError(), RuntimeError("out of memory"), ValueError("bad weights"), OSError("bad config")):
            loader = Mock(side_effect=error)
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                load_cached(loader)
            loader.assert_called_once()

    def test_gguf_patches_are_idempotent_and_preserve_quantization_metadata(self):
        from diffusers.models import model_loading_utils
        from diffusers.quantizers.gguf.utils import GGUFParameter
        from gguf import GGMLQuantizationType

        apply_gguf_patches()
        loader, constructor = model_loading_utils.load_gguf_checkpoint, GGUFParameter.__new__
        apply_gguf_patches()
        self.assertIs(loader, model_loading_utils.load_gguf_checkpoint)
        self.assertIs(constructor, GGUFParameter.__new__)
        parameter = GGUFParameter(torch.zeros((1, 34), dtype=torch.uint8), quant_type=GGMLQuantizationType.Q8_0)
        copied = GGUFParameter(parameter)
        self.assertEqual(copied.quant_type, GGMLQuantizationType.Q8_0)
        self.assertEqual(copied.quant_shape, parameter.quant_shape)


if __name__ == "__main__":
    unittest.main()
