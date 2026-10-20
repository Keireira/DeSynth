"""CPU-only regressions for preview scheduling, retention and metrics."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

import compare
import gui


class MetricsTests(unittest.TestCase):
    def test_basic_metrics_match_full_resolution_reference_without_extra_analysis(self):
        rng = np.random.default_rng(12)
        a = rng.integers(0, 256, (311, 263, 3), dtype=np.uint8)
        b = np.clip(a.astype(np.int16) + rng.integers(-20, 21, a.shape), 0, 255).astype(np.uint8)
        reference = structural_similarity(a.astype(np.float32), b.astype(np.float32),
                                          channel_axis=2, data_range=255)
        with patch.object(compare.cv2, "GaussianBlur", side_effect=AssertionError("extended metric")), \
             patch.object(compare.cv2, "calcHist", side_effect=AssertionError("extended metric")), \
             patch.object(compare, "ssim", wraps=compare.ssim) as score:
            result = compare.basic_metrics(a, b)
        delta = a.astype(np.float64) - b
        self.assertAlmostEqual(result["ssim"], float(reference), places=6)
        self.assertAlmostEqual(result["mse"], float(np.square(delta).mean()), places=10)
        self.assertAlmostEqual(result["mae"], float(np.abs(delta).mean()), places=10)
        self.assertNotIn("ssim_low_pass", result)
        self.assertTrue(all(call.args[0].shape[0] <= 134 for call in score.call_args_list))
        self.assertTrue(all(not call.kwargs.get("full", False) for call in score.call_args_list))

    def test_extended_metrics_reuse_basic_ssim_once(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        with patch.object(compare, "_ssim_rgb", wraps=compare._ssim_rgb) as score:
            result = compare.metrics(image, image.copy())
        score.assert_called_once()
        self.assertEqual(result["ssim"], 1)
        self.assertEqual(result["ssim_low_pass"], 1)
        self.assertEqual(result["ssim_high_pass"], 1)
        self.assertEqual(result["mean_hist_corr"], 1)


class StudioPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="desynth-preview-")
        self.root = Path(self.directory.name)
        self.patches = (
            patch.object(gui, "ROOT", self.root),
            patch.object(gui, "STORE", self.root / "out" / "gui"),
            patch.object(gui.Studio, "start_runtime_probe"),
        )
        for item in self.patches:
            item.start()
        self.studio = gui.Studio()
        self.source = self.studio.save(Image.new("RGBA", (16, 16), (20, 30, 40, 17)),
                                       "source", "original.png", temporary=True)
        self.clean = self.studio.save(Image.new("RGBA", (16, 16), (50, 60, 70, 255)),
                                      "clean", "clean.png", temporary=True)
        self.studio.clean_sources[self.clean["id"]] = self.source["id"]

    def tearDown(self):
        self.studio.close()
        for item in reversed(self.patches):
            item.stop()
        self.directory.cleanup()

    def payload(self, revision, **params):
        return dict(client_id="test-session", revision=revision,
                    source_id=self.source["id"], clean_id=self.clean["id"],
                    params={"restore": False, **params})

    def wait_pending(self, revision):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with self.studio.lock:
                ticket = self.studio.restore_pending.get("test-session")
                if ticket and ticket["revision"] == revision:
                    return
            time.sleep(.002)
        self.fail(f"Revision {revision} did not reach the pending slot.")

    def test_only_running_and_newest_pending_restore_execute(self):
        entered, release = threading.Event(), threading.Event()
        calls = []

        def restore(clean, original, params):
            calls.append(params["unsharp"])
            if len(calls) == 1:
                entered.set()
                self.assertTrue(release.wait(3))
            return gui.finish_image(clean, original, params, None)

        with patch.object(self.studio, "restore_image", side_effect=restore), \
             patch.object(self.studio, "compare_images", wraps=self.studio.compare_images) as metrics, \
             ThreadPoolExecutor(max_workers=3) as executor:
            first = executor.submit(self.studio.quick_restore, self.payload(1, unsharp=1))
            self.assertTrue(entered.wait(2))
            second = executor.submit(self.studio.quick_restore, self.payload(2, unsharp=2))
            self.wait_pending(2)
            third = executor.submit(self.studio.quick_restore, self.payload(3, unsharp=3))
            try:
                self.assertEqual(second.result(timeout=2), {"superseded": True})
                self.wait_pending(3)
            finally:
                release.set()
            self.assertEqual(first.result(timeout=2), {"superseded": True})
            result = third.result(timeout=2)
        self.assertEqual(calls, [1, 3])
        self.assertEqual(metrics.call_count, 1)
        self.assertEqual(result["revision"], 3)
        self.assertEqual(len([a for a in self.studio.assets.values() if a["kind"] == "result"]), 1)
        self.assertEqual(self.studio.image(result["result"]["id"]).getchannel("A").getextrema(), (17, 17))

    def test_cancel_barrier_rejects_old_work_but_allows_equal_revision_restore(self):
        self.studio.cancel_restore({"client_id": "test-session", "revision": 5})
        with patch.object(self.studio, "restore_image", wraps=self.studio.restore_image) as restore:
            self.assertEqual(self.studio.quick_restore(self.payload(4)), {"superseded": True})
            result = self.studio.quick_restore(self.payload(5))
            # An out-of-order cancellation must not rewind the revision.
            self.studio.cancel_restore({"client_id": "test-session", "revision": 3})
        self.assertEqual(result["revision"], 5)
        self.assertEqual(restore.call_count, 1)
        self.assertEqual(self.studio.restore_clients["test-session"]["revision"], 5)

    def test_cancel_during_restore_skips_metrics_and_writing(self):
        entered, release = threading.Event(), threading.Event()

        def restore(clean, original, params):
            entered.set()
            release.wait(3)
            return gui.finish_image(clean, original, params, None)

        with patch.object(self.studio, "restore_image", side_effect=restore), \
             patch.object(self.studio, "compare_images") as metrics, \
             patch.object(self.studio, "save") as save, ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(self.studio.quick_restore, self.payload(1))
            self.assertTrue(entered.wait(2))
            self.studio.cancel_restore({"client_id": "test-session", "revision": 2})
            release.set()
            self.assertEqual(pending.result(timeout=2), {"superseded": True})
        metrics.assert_not_called()
        save.assert_not_called()

    def test_temporary_retention_preserves_current_pair_and_recent_results(self):
        user_file = self.root / "user-result.png"
        Image.new("RGBA", (16, 16)).save(user_file)
        user_asset = self.studio.register(user_file, "result")
        with patch.object(gui, "MAX_TEMP_ASSETS", 5), patch.object(gui, "MAX_ASSETS", 7):
            results = [self.studio.quick_restore(self.payload(revision))["result"] for revision in range(1, 15)]
            self.assertLessEqual(len(self.studio.temporary_assets), 5)
            self.assertLessEqual(len(self.studio.assets), 7)
            self.assertEqual(len(list((gui.STORE / "temporary").glob("*.png"))), 5)
            self.studio.asset(self.source["id"])
            self.studio.asset(self.clean["id"])
            self.studio.asset(results[-1]["id"])
            self.studio.asset(results[-2]["id"])
            with self.assertRaises(gui.APIError):
                self.studio.asset(results[0]["id"])
        self.studio.forget_asset(user_asset["id"])
        self.assertTrue(user_file.is_file())
        self.studio.close()
        self.assertEqual(list((gui.STORE / "temporary").glob("*.png")), [])
        self.assertTrue(user_file.is_file())

    def test_startup_ignores_comparison_diff_and_mismatched_results(self):
        Image.new("RGBA", (20, 30)).save(self.root / "original.png")
        output = self.root / "out"
        for name, size in (("Original_desynth_valid.PNG", (20, 30)),
                           ("original_desynth_diff_x8.png", (20, 30)),
                           ("original_desynth_comparison.png", (42, 30)),
                           ("original_desynth_bad_dimensions.png", (100, 100))):
            Image.new("RGBA", size).save(output / name)
        manifest_result = output / "recorded-final.png"
        Image.new("RGBA", (20, 30)).save(manifest_result)
        (output / "original_desynth_record.json").write_text(json.dumps({
            "source": {"path": str(self.root / "original.png")},
            "outputs": [{"kind": "result", "path": str(manifest_result)}],
        }), encoding="utf-8")
        other = gui.Studio()
        try:
            self.assertEqual({a["name"] for a in other.assets.values() if a["kind"] == "result"},
                             {"Original_desynth_valid.PNG", "recorded-final.png"})
            selected = other.assets[other.default_result]
            self.assertEqual((selected["width"], selected["height"]), (20, 30))
        finally:
            other.close()

    def test_keep_intermediate_controls_clean_file_persistence(self):
        class Tensor:
            dtype = "float32"
            def is_floating_point(self):
                return True
            def to(self, **_kwargs):
                return self

        model = self.root / "Q4.gguf"
        model.write_bytes(b"model")
        adapter = self.root / "Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors"
        adapter.write_bytes(b"adapter")
        self.studio.embeds = {key: Tensor() for key in (
            "prompt_embeds", "negative_prompt_embeds", "prompt_embeds_mask", "negative_prompt_embeds_mask")}
        self.studio.module = SimpleNamespace(
            torch=SimpleNamespace(float32="float32", inference_mode=nullcontext),
            build_pipeline=lambda *_args, **_kwargs: SimpleNamespace(),
            _sample=lambda _pipe, image, _embeds, **_kwargs: image,
            _is_oom=lambda _error: False,
        )
        paths = []
        for index, keep in enumerate((False, True)):
            job_id = str(index)
            params = gui.parameters({"restore": False, "keep_intermediate": keep, "seed": 42}, process=True)
            params.update(lightning=adapter.name, _model_candidates=[model.name])
            self.studio.active = job_id
            self.studio.jobs[job_id] = dict(id=job_id, source_id=self.source["id"], runtime={
                "device": "cpu", "offload": "none", "precision": "float32"})
            self.studio.started[job_id] = time.monotonic()
            self.studio.work(job_id, self.source["id"], params)
            job = self.studio.jobs[job_id]
            self.assertEqual(job["status"], "done", job.get("error"))
            clean_id = job["clean"]["id"]
            self.assertEqual(clean_id in self.studio.temporary_assets, not keep)
            self.assertEqual(self.studio.clean_sources[clean_id], self.source["id"])
            paths.append(self.studio.paths[clean_id])
        self.studio.close()
        self.assertFalse(paths[0].exists())
        self.assertTrue(paths[1].exists())

    def test_failed_inference_releases_traceback_before_cleanup_or_fallback(self):
        class Tensor:
            dtype = "float32"
            def is_floating_point(self):
                return True
            def to(self, **_kwargs):
                return self

        class Resident:
            pass

        adapter = self.root / "adapter.safetensors"
        adapter.write_bytes(b"adapter")
        for name in ("large.gguf", "small.gguf"):
            (self.root / name).write_bytes(b"model")
        self.studio.embeds = {key: Tensor() for key in (
            "prompt_embeds", "negative_prompt_embeds", "prompt_embeds_mask", "negative_prompt_embeds_mask")}
        for error in (MemoryError("test OOM"), RuntimeError("test invalid inference")):
            with self.subTest(error=type(error).__name__):
                calls, residents, pipelines = [], [], []
                # Store only the error class/message: retaining a raised
                # exception here would itself retain the traceback under test.
                error_type, error_message = type(error), str(error)

                def build(path, **_kwargs):
                    calls.append("build " + path.name)
                    pipeline = Resident()
                    pipelines.append(weakref.ref(pipeline))
                    return pipeline

                def sample(_pipe, image, _embeds, **_kwargs):
                    calls.append("sample")
                    if len(pipelines) == 1:
                        held_by_failed_frame = Resident()
                        residents.append(weakref.ref(held_by_failed_frame))
                        raise error_type(error_message)
                    return image

                def free():
                    calls.append("free")
                    self.assertIsNone(sys.exception())
                    gc.collect()
                    self.assertIsNone(residents[0]())
                    self.assertIsNone(pipelines[0]())

                self.studio.pipe = self.studio.pipe_key = self.studio.pipe_scale = None
                self.studio.module = SimpleNamespace(
                    torch=SimpleNamespace(float32="float32", inference_mode=nullcontext),
                    build_pipeline=build, _sample=sample, _free_pipeline=free,
                    _is_oom=lambda exc: isinstance(exc, MemoryError),
                )
                job_id = error_type.__name__
                params = gui.parameters({"restore": False, "seed": 42}, process=True)
                params.update(lightning=adapter.name, _model_candidates=["large.gguf", "small.gguf"])
                self.studio.active = job_id
                self.studio.jobs[job_id] = dict(id=job_id, source_id=self.source["id"], runtime={
                    "device": "cpu", "offload": "none", "precision": "float32"})
                self.studio.started[job_id] = time.monotonic()
                with patch.object(gui.traceback, "print_exc"):
                    self.studio.work(job_id, self.source["id"], params)
                job = self.studio.jobs[job_id]
                if error_type is MemoryError:
                    self.assertEqual(job["status"], "done", job.get("error"))
                    self.assertEqual(calls, ["build large.gguf", "sample", "free", "build small.gguf", "sample"])
                else:
                    self.assertEqual(job["status"], "error")
                    self.assertEqual(job["error"], error_message)
                    self.assertEqual(calls, ["build large.gguf", "sample", "free"])
                    self.assertIsNone(self.studio.pipe)
                self.assertIsNone(self.studio.active)


if __name__ == "__main__":
    unittest.main()
