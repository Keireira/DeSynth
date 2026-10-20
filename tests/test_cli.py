"""CLI regressions using small RGBA images and a fake inference pipeline."""
from __future__ import annotations

from contextlib import nullcontext, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

from PIL import Image

import desynth as ds
from diffusers_compat import apply_gguf_patches, load_cached
from processing import finish_image, load_rgba, prepare_image, validate_restoration


class ImageProcessingTests(unittest.TestCase):
    def test_preparation_keeps_aspect_and_finish_retains_every_alpha_value(self):
        source = Image.new("RGBA", (35, 29), (80, 90, 100, 255))
        source.putalpha(Image.frombytes("L", source.size, bytes(index % 256 for index in range(35 * 29))))
        prepared = prepare_image(source)
        self.assertEqual(prepared.size, (32, 16))
        self.assertEqual(prepared.mode, "RGB")
        params = dict(restore=False, restore_sigma=1.95, restore_mode="gaussian", unsharp=0)
        restore = Mock()
        finished = finish_image(prepared, source, params, restore)
        self.assertEqual(finished.size, source.size)
        self.assertEqual(finished.mode, "RGBA")
        self.assertEqual(finished.getchannel("A").tobytes(), source.getchannel("A").tobytes())
        restore.assert_not_called()

    def test_preview_and_full_size_use_the_same_alignment(self):
        source = Image.new("RGBA", (1460, 1078))
        self.assertEqual(prepare_image(source).size, (1456, 1072))
        self.assertEqual(prepare_image(source, preview=True).size, (768, 560))

    def test_load_applies_exif_orientation_before_returning_rgba(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotated.jpg"
            source = Image.new("RGB", (35, 29))
            exif = Image.Exif()
            exif[274] = 6
            source.save(path, exif=exif)
            loaded = load_rgba(path)
            self.assertEqual(loaded.mode, "RGBA")
            self.assertEqual(loaded.size, (29, 35))

    def test_invalid_restore_values_fail_before_opencv(self):
        for sigma, mode, unsharp in (
            (float("nan"), "gaussian", 0), (float("inf"), "gaussian", 0),
            (0, "gaussian", 0), (11, "gaussian", 0), (1.95, "unknown", 0),
            (1.95, "edge", -1), (1.95, "edge", float("nan")), (1.95, "edge", 4),
        ):
            with self.subTest(values=(sigma, mode, unsharp)), self.assertRaises(ValueError):
                validate_restoration(sigma, mode, unsharp)


class CLITests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="desynth-cli-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.ai = self.root / "ai"
        self.ai.mkdir()
        self.out = self.root / "out"
        self.source_path = self.root / "original.png"
        self.source = Image.new("RGBA", (35, 29), (60, 90, 120, 255))
        self.source.putalpha(Image.frombytes("L", self.source.size, bytes(i % 256 for i in range(35 * 29))))
        self.source.save(self.source_path)
        self.model4 = self.ai / "Qwen-Image-2512-Q4_K_M.gguf"
        self.model8 = self.ai / "Qwen-Image-2512-Q8_0.gguf"
        self.model2 = self.ai / "Qwen-Image-2512-Q2_K_M.gguf"
        self.adapter = self.ai / "Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors"
        for path, size in ((self.model4, 8), (self.model8, 16), (self.model2, 4), (self.adapter, 1)):
            path.write_bytes(b"x" * size)
        for attribute, value in (
            ("ROOT", self.root), ("AI_DIR", self.ai), ("OUT_DIR", self.out),
            ("LIGHTNING_LORA", self.adapter), ("DEVICE", "cpu"), ("COMPUTE_DTYPE", ds.torch.float32),
        ):
            self.start_patch(patch.object(ds, attribute, value))
        cached = {
            "prompt_embeds": ds.torch.ones((2, 3)), "negative_prompt_embeds": ds.torch.ones((2, 3)),
            "prompt_embeds_mask": ds.torch.ones(2, dtype=ds.torch.int64),
            "negative_prompt_embeds_mask": ds.torch.ones(2, dtype=ds.torch.int64), "meta": {"test": True},
        }
        self.start_patch(patch.object(ds.torch, "load", return_value=cached))
        self.release = self.start_patch(patch.object(ds, "_free_pipeline"))
        self.samples = []
        samples = self.samples

        class Pipeline:
            def __call__(self, **kwargs):
                samples.append(kwargs)
                return SimpleNamespace(images=[kwargs["image"].copy()])

        self.pipeline_type = Pipeline
        self.build = self.start_patch(patch.object(ds, "build_pipeline", side_effect=lambda **_: Pipeline()))

    def start_patch(self, item):
        result = item.start()
        self.addCleanup(item.stop)
        return result

    def cli(self, *args):
        with patch.object(sys, "argv", ["desynth.py", str(self.source_path), "--seed", "42", *args]):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                ds.main()

    def manifests(self):
        return [json.loads(path.read_text(encoding="utf-8")) for path in self.out.rglob("*.json")]

    def assert_rgba_output(self, path):
        with Image.open(path) as result:
            self.assertEqual(result.mode, "RGBA")
            self.assertEqual(result.size, self.source.size)
            self.assertEqual(result.getchannel("A").tobytes(), self.source.getchannel("A").tobytes())

    def test_first_successful_result_is_saved_without_repeating_inference(self):
        self.cli("--passes", "3", "--denoise", ".25", ".5", "--keep-intermediate")
        self.assertEqual(len(self.samples), 6)
        self.assertEqual([call["generator"].initial_seed() for call in self.samples], [42, 43, 44, 42, 43, 44])
        self.assertTrue(all((call["width"], call["height"]) == (32, 16) for call in self.samples))
        manifest, = self.manifests()
        self.assertEqual(manifest["working_size"], [32, 16])
        self.assertEqual(manifest["transformer"]["path"], str(self.model4))
        self.assertEqual(manifest["lightning"]["path"], str(self.adapter))
        self.assertEqual(manifest["params"]["passes"], 3)
        self.assertEqual([output["denoise"] for output in manifest["outputs"]], [.25, .5])
        for output in manifest["outputs"]:
            self.assertEqual(output["kind"], "result")
            self.assert_rgba_output(output["path"])
            self.assert_rgba_output(output["clean_path"])

    def test_repeated_runs_and_models_never_overwrite_outputs(self):
        self.cli("--transformer", str(self.model4), "--no-restore")
        first = self.manifests()[0]["outputs"][0]["path"]
        first_bytes = Path(first).read_bytes()
        self.cli("--transformer", str(self.model4), "--no-restore")
        self.cli("--transformer", str(self.model8), "--seed", "43", "--no-restore")
        manifests = self.manifests()
        self.assertEqual(len(manifests), 3)
        paths = [manifest["outputs"][0]["path"] for manifest in manifests]
        self.assertEqual(len(set(paths)), 3)
        self.assertEqual(Path(first).read_bytes(), first_bytes)
        self.assertTrue(any("q8_0" in path and "seed43" in path for path in paths))
        for path in paths:
            self.assert_rgba_output(path)

    def test_duplicate_strengths_in_one_sweep_have_distinct_files(self):
        self.cli("--denoise", ".25", ".25", "--no-restore")
        manifest, = self.manifests()
        self.assertEqual(len(self.samples), 2)
        self.assertEqual(len({item["path"] for item in manifest["outputs"]}), 2)

    def test_reference_exports_only_final_and_comparison_with_short_names(self):
        self.cli("--steps", "32", "--denoise", ".125", ".25", "--keep-intermediate")
        manifest, = self.manifests()
        expected = {
            f"{prefix}Original_desynth_s32_d{strength}_p1_r1.95.png"
            for strength in ("0.125", "0.250") for prefix in ("", "comparison_")
        }
        self.assertEqual({path.name for path in self.out.iterdir() if path.is_file()}, expected)
        archive = self.out / "runs" / manifest["run_id"]
        for output in manifest["outputs"]:
            self.assertEqual(Path(output["path"]).parent, archive)
            self.assertEqual(Path(output["clean_path"]).parent, archive)
            self.assertEqual(Path(output["comparison_path"]).parent, archive)
            self.assertEqual(Path(output["reference_path"]).read_bytes(), Path(output["path"]).read_bytes())
            self.assert_rgba_output(output["reference_path"])
            with Image.open(output["reference_comparison_path"]) as comparison:
                self.assertEqual(comparison.mode, "RGB")

    def test_edge_and_sharpen_reference_names_describe_restoration(self):
        self.cli("--restore-mode", "edge", "--unsharp", ".2")
        output = self.manifests()[0]["outputs"][0]
        self.assertEqual(Path(output["reference_path"]).name,
                         "Original_desynth_s32_d0.250_p1_r1.95_edge_u0.2.png")

    def test_other_image_named_original_does_not_create_reference_exports(self):
        other = self.root / "elsewhere" / "original.png"
        other.parent.mkdir()
        self.source.save(other)
        self.source_path = other
        self.cli("--keep-intermediate")
        manifest, = self.manifests()
        self.assertEqual([path for path in self.out.iterdir() if path.is_file()], [])
        output = manifest["outputs"][0]
        self.assertNotIn("reference_path", output)
        self.assert_rgba_output(output["path"])
        self.assert_rgba_output(output["clean_path"])

    def test_reference_alias_is_updated_while_archive_stays_immutable(self):
        self.cli("--no-restore")
        first_manifest, = self.manifests()
        first = first_manifest["outputs"][0]
        archived_bytes = Path(first["path"]).read_bytes()
        reference = Path(first["reference_path"])
        self.assertIn("_norestore.png", reference.name)
        self.source = Image.new("RGBA", self.source.size, (200, 40, 50, 128))
        self.source.save(self.source_path)
        self.cli("--no-restore")
        self.assertEqual(Path(first["path"]).read_bytes(), archived_bytes)
        self.assertNotEqual(reference.read_bytes(), archived_bytes)
        self.assertEqual(len(self.manifests()), 2)
        self.assertEqual(len(list(self.out.glob("*.png"))), 2)

    def test_oom_releases_failed_pipeline_before_loading_smaller_model(self):
        refs, release_checks, loaded = [], [], []
        samples = self.samples

        class FailingPipeline:
            def __call__(self, **kwargs):
                samples.append(kwargs)
                raise RuntimeError("out of memory")

        def build(**kwargs):
            loaded.append(kwargs["transformer_path"])
            pipe = FailingPipeline() if len(loaded) == 1 else self.pipeline_type()
            refs.append(weakref.ref(pipe))
            return pipe

        self.build.side_effect = build
        self.release.side_effect = lambda: release_checks.append([ref() is None for ref in refs])
        self.cli("--no-restore")
        self.assertEqual(loaded, [self.model4, self.model2])
        self.assertEqual(release_checks[0], [True])
        self.assertEqual(len(self.samples), 2)
        manifest, = self.manifests()
        self.assertEqual(manifest["transformer"]["path"], str(self.model2))

    def test_explicit_model_oom_does_not_switch_models(self):
        self.build.side_effect = RuntimeError("out of memory")
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            self.cli("--transformer", str(self.model8))
        self.build.assert_called_once()

    def test_invalid_options_rejected_before_model_loading(self):
        for args in (("--restore-sigma", "nan"), ("--restore-sigma", "0"), ("--unsharp", "-1"),
                     ("--unsharp", "inf"), ("--denoise", "nan"), ("--seed", "-1")):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                self.cli(*args)
        self.build.assert_not_called()


class CompatibilityTests(unittest.TestCase):
    def test_cleanup_covers_each_cuda_device_after_a_runtime_switch(self):
        with (patch.object(ds.torch.cuda, "is_available", return_value=True),
              patch.object(ds.torch.cuda, "device_count", return_value=2),
              patch.object(ds.torch.cuda, "device", side_effect=lambda _: nullcontext()) as device,
              patch.object(ds.torch.cuda, "empty_cache") as empty,
              patch.object(ds.torch.cuda, "ipc_collect") as collect,
              patch.object(ds.torch.backends.mps, "is_available", return_value=False)):
            ds._free_pipeline()
        self.assertEqual([call.args[0] for call in device.call_args_list], [0, 1])
        self.assertEqual(empty.call_count, 2)
        self.assertEqual(collect.call_count, 2)

    def test_lora_error_is_not_retried_with_another_loader(self):
        pipeline = Mock()
        pipeline.load_lora_weights.side_effect = RuntimeError("out of memory")
        with tempfile.TemporaryDirectory() as directory:
            model, adapter, embeds = [Path(directory) / name for name in ("model.gguf", "adapter.safetensors", "embeds.pt")]
            for path in (model, adapter, embeds):
                path.write_bytes(b"test")
            with (
                patch.object(ds, "EMBEDS_CACHE", embeds),
                patch.object(ds.QwenImageTransformer2DModel, "from_single_file", return_value=Mock()),
                patch.object(ds.AutoencoderKLQwenImage, "from_pretrained", return_value=Mock()),
                patch.object(ds.QwenImageImg2ImgPipeline, "from_pretrained", return_value=pipeline),
                patch("safetensors.torch.load_file", return_value={"weight": object()}) as load_weights,
            ):
                with self.assertRaisesRegex(RuntimeError, "out of memory"):
                    ds.build_pipeline(model, lightning_lora=adapter, device="cpu", offload="none")
                load_weights.assert_called_once_with(str(adapter), device="cpu")
                pipeline.load_lora_weights.assert_called_once()

    def test_cache_miss_retries_once_but_loading_failures_do_not(self):
        from huggingface_hub.errors import LocalEntryNotFoundError

        miss = LocalEntryNotFoundError("Not cached")
        wrapper = OSError("Diffusers wrapper")
        wrapper.__cause__ = miss
        loader = Mock(side_effect=[wrapper, "loaded"])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(load_cached(loader, config="local"), "loaded")
        self.assertEqual([call.kwargs["local_files_only"] for call in loader.call_args_list], [True, False])
        for error in (MemoryError(), RuntimeError("out of memory"), ValueError("bad weights"), OSError("bad config")):
            loader = Mock(side_effect=error)
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                load_cached(loader)
            loader.assert_called_once()

    def test_patches_are_idempotent_and_preserve_quant_metadata(self):
        from diffusers.models import model_loading_utils
        from diffusers.quantizers.gguf.utils import GGUFParameter
        from gguf import GGMLQuantizationType

        apply_gguf_patches()
        loader, constructor = model_loading_utils.load_gguf_checkpoint, GGUFParameter.__new__
        apply_gguf_patches()
        self.assertIs(loader, model_loading_utils.load_gguf_checkpoint)
        self.assertIs(constructor, GGUFParameter.__new__)
        parameter = GGUFParameter(ds.torch.zeros((1, 34), dtype=ds.torch.uint8), quant_type=GGMLQuantizationType.Q8_0)
        copied = GGUFParameter(parameter)
        self.assertEqual(copied.quant_type, GGMLQuantizationType.Q8_0)
        self.assertEqual(copied.quant_shape, parameter.quant_shape)


if __name__ == "__main__":
    unittest.main()
