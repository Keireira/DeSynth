"""Local HTTP regression tests; no model loading, CUDA, or external network."""
from __future__ import annotations

import base64
from contextlib import nullcontext
import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

import gui


class QuietHandler(gui.Handler):
    def log_message(self, *_args):
        pass


class GuiHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="desynth-gui-test-")
        cls.root = Path(cls.temporary.name)
        (cls.root / "ai").mkdir()
        for steps in (4, 8):
            (cls.root / "ai" / f"Qwen-Image-2512-Lightning-{steps}steps-V1.0-fp32.safetensors").write_bytes(b"test adapter")
        (cls.root / "test.gguf").write_bytes(b"test transformer")
        (cls.root / "embeds_cache.pt").write_bytes(b"test embeddings")
        cls.patches = (
            patch.object(gui, "ROOT", cls.root),
            patch.object(gui, "STORE", cls.root / "out" / "gui"),
            patch.object(gui.Studio, "probe_device", lambda self: None),
        )
        for item in cls.patches:
            item.start()
        cls.server = gui.LocalServer(("127.0.0.1", 0), QuietHandler)
        cls.thread = threading.Thread(
            target=cls.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True
        )
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        for item in reversed(cls.patches):
            item.stop()
        cls.temporary.cleanup()

    def setUp(self):
        self.studio = self.server.studio = gui.Studio()
        self.connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        # Device discovery is disabled above: tests must not query the real GPU.
        self.studio.runtime = {
            "ready": True,
            "default_device": "cpu",
            "devices": [{
                "id": "cpu", "name": "CPU", "type": "cpu",
                "precisions": ["float32"], "offload_modes": ["none"],
            }],
        }

    def tearDown(self):
        self.connection.close()

    def add_fake_cuda_devices(self):
        for index in (0, 1):
            self.studio.runtime["devices"].append({
                "id": f"cuda:{index}", "name": f"Test GPU {index}", "type": "cuda",
                "precisions": ["bfloat16", "float32"],
                "offload_modes": ["sequential", "model", "none"],
            })
        self.studio.runtime["default_device"] = "cuda:0"

    def request(self, path, payload=None, headers=None):
        request_headers = {}
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        request_headers.update(headers or {})
        self.connection.request("POST" if payload is not None else "GET", path, body, request_headers)
        response = self.connection.getresponse()
        content = response.read()
        if "application/json" in response.getheader("Content-Type", ""):
            content = json.loads(content)
        return response.status, content, dict(response.getheaders())

    @staticmethod
    def image_payload(image, role="source", name="image.png", format="PNG", **save_options):
        data = io.BytesIO()
        image.save(data, format=format, **save_options)
        return {
            "role": role,
            "name": name,
            "data": "data:image/" + format.lower() + ";base64," + base64.b64encode(data.getvalue()).decode("ascii"),
        }

    def upload(self, image, role="source", **options):
        status, asset, _headers = self.request("/api/upload", self.image_payload(image, role, **options))
        self.assertEqual(status, 200, asset)
        return asset

    def wait_for_job(self, job_id):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status, job, _headers = self.request("/api/job/" + job_id)
            self.assertEqual(status, 200)
            if job["status"] in ("done", "error") and self.studio.active is None:
                return job
            time.sleep(.01)
        self.fail("Worker did not finish within three seconds")

    def test_rgba_pixels_preserve_colors_under_zero_and_partial_alpha(self):
        image = Image.new("RGBA", (2, 1))
        image.putdata([(255, 0, 173, 0), (7, 199, 53, 17)])
        asset = self.upload(image)
        status, pixels, headers = self.request("/api/pixels/" + asset["id"])
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/octet-stream")
        self.assertEqual(pixels, image.tobytes())
        self.assertEqual(len(pixels), 2 * 1 * 4)

    def test_upload_applies_exif_orientation_and_normalizes_to_rgba_png(self):
        image = Image.new("RGB", (2, 3), (21, 45, 76))
        exif = image.getexif()
        exif[274] = 6  # JPEG orientation: rotate 90 degrees clockwise.
        asset = self.upload(image, name="../../outside.jpg", format="JPEG", exif=exif)
        self.assertEqual((asset["width"], asset["height"]), (3, 2))
        self.assertEqual(asset["name"], "outside.jpg")
        status, data, headers = self.request(asset["url"])
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/png")
        with Image.open(io.BytesIO(data)) as normalized:
            self.assertEqual(normalized.format, "PNG")
            self.assertEqual(normalized.mode, "RGBA")
            self.assertEqual(normalized.size, (3, 2))
            self.assertEqual(normalized.getchannel("A").getextrema(), (255, 255))
        self.assertTrue(self.studio.paths[asset["id"]].is_relative_to(gui.STORE))

    def test_compare_detects_alpha_only_and_hidden_rgb_changes(self):
        original = Image.new("RGBA", (2, 2), (20, 40, 60, 0))
        changed = original.copy()
        changed.putpixel((0, 0), (20, 40, 60, 12))
        source = self.upload(original)
        result = self.upload(changed, "result")
        status, metrics, _headers = self.request("/api/compare", {
            "source_id": source["id"], "result_id": result["id"],
        })
        self.assertEqual(status, 200, metrics)
        self.assertFalse(metrics["identical"])
        self.assertEqual(metrics["changed_pct"], 25)
        self.assertEqual(metrics["mean_alpha"], 3)
        self.assertEqual(metrics["mae"], 0)

        hidden_change = original.copy()
        hidden_change.putpixel((0, 0), (255, 0, 0, 0))
        result = self.upload(hidden_change, "result")
        status, metrics, _headers = self.request("/api/compare", {
            "source_id": source["id"], "result_id": result["id"],
        })
        self.assertEqual(status, 200)
        self.assertEqual(metrics["changed_pct"], 25)
        self.assertEqual(metrics["mean_alpha"], 0)
        self.assertGreater(metrics["mae"], 0)

    def test_compare_rejects_different_dimensions(self):
        source = self.upload(Image.new("RGBA", (2, 2)))
        result = self.upload(Image.new("RGBA", (3, 2)), "result")
        status, error, _headers = self.request("/api/compare", {
            "source_id": source["id"], "result_id": result["id"],
        })
        self.assertEqual(status, 400)
        self.assertIn("dimensions", error["error"])

    def test_invalid_processing_parameters_never_start_a_job(self):
        source = self.upload(Image.new("RGBA", (16, 16)))
        for params in ({"denoise": True}, {"denoise": float("nan")},
                       {"steps": 8.5}, {"passes": 9}, {"seed": -1},
                       {"restore_mode": "unknown"}, {"restore": "yes"}):
            with self.subTest(params=params):
                status, error, _headers = self.request("/api/process", {
                    "source_id": source["id"], "params": params,
                })
                self.assertEqual(status, 400, error)
                self.assertIn("error", error)
        self.assertFalse(self.studio.jobs)
        self.assertIsNone(self.studio.module)
        self.assertIsNone(self.studio.active)

    def test_cross_origin_requests_and_path_traversal_are_rejected(self):
        payload = self.image_payload(Image.new("RGBA", (1, 1)))
        status, _error, _headers = self.request("/api/upload", payload, {"Origin": "https://example.invalid"})
        self.assertEqual(status, 403)
        status, _error, _headers = self.request("/api/upload", payload, {"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(status, 403)
        self.assertFalse(self.studio.assets)
        for path in ("/../gui.py", "/api/image/../../gui.py", "/api/pixels/../../gui.py"):
            with self.subTest(path=path):
                status, error, _headers = self.request(path)
                self.assertEqual(status, 404)
                self.assertIn("error", error)

    def test_state_reports_only_the_detected_runtime_capabilities(self):
        status, session, _headers = self.request("/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(session["runtime"], self.studio.runtime)
        self.assertEqual(session["runtime"]["default_device"], "cpu")
        self.assertEqual([item["id"] for item in session["runtime"]["devices"]], ["cpu"])

    def test_state_discovers_ai_adapters_and_prefers_the_8_step_profile(self):
        status, session, _headers = self.request("/api/state")
        self.assertEqual(status, 200)
        self.assertTrue(session["ready"], session["missing"])
        self.assertEqual([item["steps"] for item in session["lightning_adapters"]], [8, 4])
        self.assertEqual(session["default_lightning"], "ai/Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors")
        self.assertEqual(session["default_steps"], 32)

    def test_automatic_model_fallback_never_tries_a_larger_quant(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = [{"path": f"Q{bits}.gguf"} for bits in (4, 2, 8, 3)]
            for bits in (4, 2, 8, 3):
                (root / f"Q{bits}.gguf").write_bytes(b"x" * bits)
            self.assertEqual(gui.automatic_model_paths(models, root), ["Q4.gguf", "Q3.gguf", "Q2.gguf"])
            self.assertEqual(gui.automatic_model_paths([models[2]], root), ["Q8.gguf"])

    def test_invalid_adapter_paths_and_scales_never_start_a_job(self):
        source = self.upload(Image.new("RGBA", (16, 16)))
        invalid = (
            {"lightning": "../outside.safetensors"},
            {"lightning": "missing.safetensors"},
            {"lightning": ["not a path"]},
            {"lightning_scale": 0}, {"lightning_scale": -1},
            {"lightning_scale": True}, {"lightning_scale": float("nan")},
        )
        for params in invalid:
            with self.subTest(params=params):
                status, error, _headers = self.request("/api/process", {"source_id": source["id"], "params": params})
                self.assertEqual(status, 400, error)
        self.assertFalse(self.studio.jobs)
        self.assertIsNone(self.studio.module)

    def test_adapter_profile_defaults_do_not_override_explicit_steps(self):
        source = self.upload(Image.new("RGBA", (16, 16)))
        completed = threading.Event()

        def finish_without_loading(*_args):
            self.studio.active = None
            completed.set()

        with patch.object(self.studio, "work", side_effect=finish_without_loading):
            for profile, explicit, expected in ((8, None, 32), (4, None, 8), (8, 12, 12)):
                completed.clear()
                adapter = f"ai/Qwen-Image-2512-Lightning-{profile}steps-V1.0-fp32.safetensors"
                params = {"lightning": adapter}
                if explicit is not None:
                    params["steps"] = explicit
                status, job, _headers = self.request("/api/process", {"source_id": source["id"], "params": params})
                self.assertEqual(status, 202, job)
                self.assertTrue(completed.wait(1))
                self.assertEqual(job["lightning"], adapter)
                self.assertEqual(job["steps"], expected)

    def test_unavailable_devices_and_unsupported_cpu_options_do_not_start_jobs(self):
        source = self.upload(Image.new("RGBA", (16, 16)))
        invalid = (
            {"device": "cuda:0"}, {"device": "cuda:99"}, {"device": "mps"},
            {"device": "cpu", "offload": "sequential"},
            {"device": "cpu", "offload": "model"},
            {"device": "cpu", "precision": "float16"},
            {"device": "cpu", "precision": "bfloat16"},
            {"device": "other"}, {"offload": "other"}, {"precision": "int8"},
        )
        with patch.object(self.studio, "state", return_value={"ready": True}):
            for params in invalid:
                with self.subTest(params=params):
                    status, error, _headers = self.request("/api/process", {
                        "source_id": source["id"], "params": params,
                    })
                    self.assertEqual(status, 400, error)
                    self.assertIn("error", error)
        self.assertFalse(self.studio.jobs)
        self.assertIsNone(self.studio.module)
        self.assertIsNone(self.studio.active)

    def test_process_records_effective_runtime_independently_of_later_device_changes(self):
        source = self.upload(Image.new("RGBA", (16, 16)))
        started = threading.Event()
        with patch.object(self.studio, "state", return_value={"ready": True}), patch.object(
            self.studio, "work", side_effect=lambda *_args: started.set()
        ):
            status, job, _headers = self.request("/api/process", {
                "source_id": source["id"],
                "params": {"device": "auto", "offload": "auto", "precision": "auto"},
            })
            self.assertEqual(status, 202, job)
            self.assertTrue(started.wait(1), "The accepted job did not reach its worker")
        self.assertEqual(job["runtime"]["device"], "cpu")
        self.assertEqual(job["runtime"]["offload"], "none")
        self.assertEqual(job["runtime"]["precision"], "float32")
        self.assertEqual(job["requested_runtime"], {"device": "auto", "offload": "auto", "precision": "auto"})
        self.add_fake_cuda_devices()
        status, saved, _headers = self.request("/api/job/" + job["id"])
        self.assertEqual(status, 200)
        self.assertEqual(saved["runtime"], job["runtime"])
        self.assertEqual(self.studio.resolve_runtime(gui.parameters({}, process=True))["device"], "cuda:0")

    def test_qwen_rejects_float16_and_uses_float32_when_bfloat16_is_unavailable(self):
        self.add_fake_cuda_devices()
        gpu = self.studio.runtime["devices"][1]
        source = self.upload(Image.new("RGBA", (16, 16)))
        with patch.object(self.studio, "state", return_value={"ready": True}):
            for precisions, expected_auto in ((["bfloat16", "float32"], "bfloat16"), (["float32"], "float32")):
                with self.subTest(precisions=precisions):
                    gpu["precisions"] = precisions
                    effective = self.studio.resolve_runtime(gui.parameters({"device": "cuda:0"}, process=True))
                    self.assertEqual(effective["precision"], expected_auto)
                    status, error, _headers = self.request("/api/process", {
                        "source_id": source["id"],
                        "params": {"device": "cuda:0", "precision": "float16"},
                    })
                    self.assertEqual(status, 400, error)
                    self.assertIn("float16", error["error"].lower())
                    self.assertIn("float32", error["error"].lower())
        self.assertFalse(self.studio.jobs)
        self.assertIsNone(self.studio.active)
        self.assertIsNone(self.studio.module)

    def test_pipeline_reuse_tracks_runtime_and_preserves_integer_embedding_masks(self):
        class FakeTensor:
            def __init__(self, dtype, device="cpu"):
                self.dtype, self.device = dtype, device

            def is_floating_point(self):
                return self.dtype in ("float32", "float16", "bfloat16")

            def to(self, *, dtype, device):
                return FakeTensor(dtype, device)

        self.add_fake_cuda_devices()
        source = self.upload(Image.new("RGBA", (2, 2), (10, 20, 30, 17)))
        original_dtypes = {
            "prompt_embeds": "float32", "negative_prompt_embeds": "float32",
            "prompt_embeds_mask": "bool", "negative_prompt_embeds_mask": "int64",
        }
        cached = {key: FakeTensor(dtype) for key, dtype in original_dtypes.items()}
        metadata = {"model": "Qwen-Image", "prompt": "", "version": 1}
        cached["meta"] = metadata
        build = Mock(side_effect=lambda *_args, **_kwargs: SimpleNamespace(num_timesteps=2))
        sample = Mock(side_effect=lambda _pipe, image, _embeds, **_kwargs: image)
        free = Mock()
        load = Mock(return_value=cached)
        self.studio.models = [{"name": "test.gguf", "path": "test.gguf"}]
        self.studio.module = SimpleNamespace(
            EMBEDS_CACHE=self.root / "unused.pt", build_pipeline=build,
            _sample=sample, _free_pipeline=free, _is_oom=lambda _error: False,
            torch=SimpleNamespace(
                load=load, float32="float32", float16="float16", bfloat16="bfloat16",
                inference_mode=nullcontext, cuda=SimpleNamespace(is_available=lambda: False),
            ),
        )
        cases = (
            ("cuda:0", "sequential", "bfloat16", 8, .8, 1),
            ("cuda:0", "sequential", "bfloat16", 8, .8, 1),  # Identical configuration reuses the pipeline.
            ("cuda:1", "sequential", "bfloat16", 8, .8, 2),  # Device alone changes.
            ("cuda:1", "model", "bfloat16", 8, .8, 3),       # Offload alone changes.
            ("cuda:1", "model", "float32", 8, .8, 4),        # Precision alone changes.
            ("cpu", "none", "float32", 8, .8, 5),
            ("cpu", "none", "float32", 4, .8, 6),          # Adapter alone changes.
            ("cpu", "none", "float32", 4, .6, 7),          # Adapter strength alone changes.
            ("cpu", "none", "float32", 4, .6, 7),
        )
        with patch.object(self.studio, "state", return_value={"ready": True}):
            for device, offload, precision, profile, scale, expected_builds in cases:
                adapter = f"ai/Qwen-Image-2512-Lightning-{profile}steps-V1.0-fp32.safetensors"
                with self.subTest(device=device, offload=offload, precision=precision, adapter=adapter, scale=scale):
                    status, accepted, _headers = self.request("/api/process", {
                        "source_id": source["id"], "params": {
                            "device": device, "offload": offload, "precision": precision,
                            "restore": False, "seed": 42,
                            "lightning": adapter, "lightning_scale": scale,
                        },
                    })
                    self.assertEqual(status, 202, accepted)
                    finished = self.wait_for_job(accepted["id"])
                    self.assertEqual(finished["status"], "done", finished.get("error"))
                    self.assertEqual(build.call_count, expected_builds)
                    self.assertEqual(self.studio.pipe_key, (
                        gui.weight_identity(self.root / "test.gguf"), gui.weight_identity(self.root / adapter),
                        scale, device, offload, precision,
                    ))
                    self.assertEqual(build.call_args.kwargs, {
                        "device": device, "offload": offload, "compute_dtype": precision,
                        "lightning_lora": self.root / adapter, "lightning_scale": scale,
                    })
                    prepared = sample.call_args.args[2]
                    self.assertNotIn("meta", prepared)
                    for key, dtype in original_dtypes.items():
                        self.assertEqual(prepared[key].device, device)
                        self.assertEqual(prepared[key].dtype, precision if "mask" not in key else dtype)
                    self.assertEqual(finished["runtime"]["device"], device)
                    self.assertEqual(finished["runtime"]["offload"], offload)
                    self.assertEqual(finished["runtime"]["precision"], precision)
        load.assert_called_once()
        self.assertEqual(free.call_count, 4)
        for key in original_dtypes:
            tensor = cached[key]
            self.assertEqual(tensor.device, "cpu")
            self.assertEqual(tensor.dtype, original_dtypes[key])
        self.assertIs(cached["meta"], metadata)
        self.assertEqual(metadata, {"model": "Qwen-Image", "prompt": "", "version": 1})

    def test_concurrent_processing_returns_conflict_without_loading_model(self):
        source = self.upload(Image.new("RGBA", (16, 16)))
        self.studio.active = "existing-job"
        with patch.object(self.studio, "state", return_value={"ready": True}):
            status, error, _headers = self.request("/api/process", {"source_id": source["id"], "params": {}})
        self.assertEqual(status, 409, error)
        self.assertEqual(self.studio.active, "existing-job")
        self.assertFalse(self.studio.jobs)
        self.assertIsNone(self.studio.module)

    def test_failed_worker_is_reported_by_job_api_and_releases_busy_state(self):
        source = self.upload(Image.new("RGBA", (16, 16)))
        fake_load = Mock(side_effect=RuntimeError("Synthetic model-load failure"))
        self.studio.module = SimpleNamespace(
            DEVICE="cpu", EMBEDS_CACHE=self.root / "unused.pt",
            torch=SimpleNamespace(load=fake_load, float32="float32"), _is_oom=lambda _error: False,
        )
        with patch.object(self.studio, "state", return_value={"ready": True}), patch.object(gui.traceback, "print_exc"):
            status, job, _headers = self.request("/api/process", {"source_id": source["id"], "params": {"seed": 42}})
            self.assertEqual(status, 202, job)
            finished = self.wait_for_job(job["id"])
        self.assertEqual(finished["status"], "error")
        self.assertEqual(finished["error"], "Synthetic model-load failure")
        self.assertEqual(finished["seed"], 42)
        self.assertIsNone(finished["result"])
        self.assertIsNone(self.studio.active)
        fake_load.assert_called_once()

    def test_quick_restore_preserves_source_alpha_and_checks_clean_ownership(self):
        original = Image.new("RGBA", (2, 2), (10, 20, 30, 17))
        original.putpixel((0, 0), (1, 2, 3, 0))
        source = self.upload(original)
        clean = self.studio.save(Image.new("RGBA", (2, 2), (70, 80, 90, 255)), "clean", "clean.png")
        self.studio.clean_sources[clean["id"]] = source["id"]
        status, restored, _headers = self.request("/api/restore", {
            "source_id": source["id"], "clean_id": clean["id"], "params": {"restore": False},
        })
        self.assertEqual(status, 200, restored)
        status, pixels, _headers = self.request("/api/pixels/" + restored["result"]["id"])
        self.assertEqual(status, 200)
        expected = Image.new("RGBA", (2, 2), (70, 80, 90, 255))
        expected.putalpha(original.getchannel("A"))
        self.assertEqual(pixels, expected.tobytes())
        other_source = self.upload(original)
        status, error, _headers = self.request("/api/restore", {
            "source_id": other_source["id"], "clean_id": clean["id"], "params": {"restore": False},
        })
        self.assertEqual(status, 400)
        self.assertIn("different source", error["error"])
        self.assertIsNone(self.studio.module)


if __name__ == "__main__":
    unittest.main()
