"""Model discovery, Lightning wiring, and offline setup regressions.

Fixtures contain placeholder bytes; no weights, inference, or network are used.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import desynth as ds
from scripts import project


GGUF4 = "Qwen-Image-2512-Q4_K_M.gguf"
GGUF8 = "Qwen-Image-2512-Q8_0.gguf"
LORA4 = "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors"
LORA8 = "Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors"
LORA8_BF16 = "Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors"


class LocalModelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="desynth-model-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ai = self.root / "ai"
        self.ai.mkdir()
        for owner, attribute, value in (
            (ds, "ROOT", self.root), (ds, "AI_DIR", self.ai),
            (project, "ROOT", self.root), (project, "MODEL_DIR", self.ai),
        ):
            self.start_patch(patch.object(owner, attribute, value))

    def start_patch(self, item):
        result = item.start()
        self.addCleanup(item.stop)
        return result

    @staticmethod
    def asset(directory, name, data=b"placeholder model"):
        path = directory / name
        path.write_bytes(data)
        return path

    def test_bare_filename_prefers_ai_but_skips_empty_files(self):
        legacy = self.asset(self.root, GGUF8)
        current = self.asset(self.ai, GGUF8)
        self.assertEqual(ds.resolve_model_path(GGUF8), current)
        current.write_bytes(b"")
        self.assertEqual(ds.resolve_model_path(GGUF8), legacy)
        self.assertEqual(ds.resolve_model_path(legacy), legacy)

    def test_gguf_discovery_deduplicates_and_retains_other_quants(self):
        q4 = self.asset(self.ai, GGUF4)
        self.asset(self.root, GGUF4)
        q8 = self.asset(self.root, GGUF8)
        self.asset(self.ai, GGUF8, b"")
        self.asset(self.ai, "empty.gguf", b"")
        self.assertEqual(ds.discover_gguf_models(), [q4, q8])

    def test_q8_only_installation_is_discoverable(self):
        q8 = self.asset(self.ai, GGUF8)
        self.assertEqual(ds.discover_gguf_models(), [q8])
        self.assertEqual(ds.resolve_model_path(GGUF8), q8)

    def test_setup_accepts_embeddings_in_ai(self):
        self.asset(self.ai, "embeds_cache.pt")
        project.require_embeds()
        self.assertEqual(project.find_asset("embeds_cache.pt"), self.ai / "embeds_cache.pt")

    def test_lightning_discovery_prefers_8_and_deduplicates_ai(self):
        lora8 = self.asset(self.ai, LORA8)
        self.asset(self.root, LORA8)
        lora4 = self.asset(self.root, LORA4)
        self.asset(self.ai, LORA4, b"")
        self.asset(self.ai, "Qwen-Image-2512-Lightning-6steps-V1.0-fp32.safetensors")
        self.asset(self.ai, "unrelated.safetensors")
        self.assertEqual(ds.discover_lightning_loras(), [lora8, lora4])

    def test_lightning_prefers_fp32_even_if_bf16_is_in_ai(self):
        bf16 = self.asset(self.ai, LORA8_BF16)
        fp32 = self.asset(self.root, LORA8)
        self.assertEqual(ds.discover_lightning_loras(), [fp32, bf16])

    def test_default_schedule_matches_selected_adapter(self):
        self.assertEqual(ds.lightning_steps(LORA8), 8)
        self.assertEqual(ds.default_steps_for_lightning(LORA8), 32)
        self.assertEqual(ds.effective_img2img_steps(32, .25), 8)
        self.assertEqual(ds.lightning_steps(LORA4), 4)
        self.assertEqual(ds.default_steps_for_lightning(LORA4), 8)
        self.assertEqual(ds.effective_img2img_steps(8, .25), 2)

    def test_effective_steps_match_installed_diffusers_rounding(self):
        for steps, strength in ((8, .275), (8, .01), (32, .25), (8, 1), (23, .49)):
            with self.subTest(steps=steps, strength=strength):
                scheduler = SimpleNamespace(
                    timesteps=ds.torch.arange(steps), order=1, set_begin_index=Mock(),
                )
                timesteps, actual = ds.QwenImageImg2ImgPipeline.get_timesteps(
                    SimpleNamespace(scheduler=scheduler), steps, strength, "cpu",
                )
                self.assertEqual(ds.effective_img2img_steps(steps, strength), actual)
                self.assertEqual(actual, len(timesteps))
        self.assertEqual(ds.effective_img2img_steps(8, .275), 3)

    def test_pipeline_uses_selected_lora_scale_and_lightning_scheduler(self):
        transformer = self.asset(self.ai, GGUF4)
        selected_lora = self.asset(self.ai, LORA8)
        legacy_lora = self.asset(self.root, LORA4)
        embeds = self.asset(self.root, "embeds_cache.pt")
        self.start_patch(patch.object(ds, "LIGHTNING_LORA", legacy_lora))
        self.start_patch(patch.object(ds, "EMBEDS_CACHE", embeds))
        transformer_loader = self.start_patch(patch.object(
            ds.QwenImageTransformer2DModel, "from_single_file", return_value=Mock(),
        ))
        self.start_patch(patch.object(ds.AutoencoderKLQwenImage, "from_pretrained", return_value=Mock()))
        pipeline = Mock()
        pipeline_loader = self.start_patch(patch.object(
            ds.QwenImageImg2ImgPipeline, "from_pretrained", return_value=pipeline,
        ))
        state = {"selected-adapter": object()}
        weights_loader = self.start_patch(patch("safetensors.torch.load_file", return_value=state))
        placement = self.start_patch(patch.object(ds, "_place_pipeline"))
        with redirect_stdout(io.StringIO()):
            result = ds.build_pipeline(
                transformer, lightning_lora=selected_lora, lightning_scale=.65,
                device="cpu", offload="none", compute_dtype=ds.torch.float32,
            )
        self.assertIs(result, pipeline)
        self.assertEqual(transformer_loader.call_args.kwargs["pretrained_model_link_or_path_or_dict"], str(transformer))
        weights_loader.assert_called_once_with(str(selected_lora), device="cpu")
        pipeline.load_lora_weights.assert_called_once_with(state, adapter_name="lightning")
        pipeline.set_adapters.assert_called_once_with(["lightning"], adapter_weights=[.65])
        placement.assert_called_once_with(pipeline, "cpu", "none")
        pipeline.vae.enable_tiling.assert_called_once_with()
        config = pipeline_loader.call_args.kwargs["scheduler"].config
        self.assertAlmostEqual(config.base_shift, math.log(3))
        self.assertAlmostEqual(config.max_shift, math.log(3))
        self.assertTrue(config.use_dynamic_shifting)
        self.assertEqual(config.time_shift_type, "exponential")
        self.assertIsNone(config.shift_terminal)

    def assert_setup_reuses_local_models(self):
        api = self.start_patch(patch("huggingface_hub.HfApi"))
        download = self.start_patch(patch("huggingface_hub.hf_hub_download"))
        api.side_effect = AssertionError("Setup must not access HF for installed models")
        download.side_effect = AssertionError("Setup must not download installed models")
        with redirect_stdout(io.StringIO()):
            project.download_models("auto")
        api.assert_not_called()
        download.assert_not_called()

    def test_setup_reuses_ai_q8_and_8step_without_network(self):
        self.asset(self.ai, GGUF8)
        self.asset(self.ai, LORA8)
        self.asset(self.root, "embeds_cache.pt")
        self.assert_setup_reuses_local_models()

    def test_setup_reuses_legacy_4step_without_network(self):
        self.asset(self.root, GGUF4)
        self.asset(self.root, LORA4)
        self.asset(self.root, "embeds_cache.pt")
        self.assert_setup_reuses_local_models()


if __name__ == "__main__":
    unittest.main()
