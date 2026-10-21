"""HTTP workflows with real image processing and deterministic CPU-only sampling."""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageFilter

import desynth
import gui


class QuietHandler(gui.Handler):
    def log_message(self, *_args):
        pass


class GuiWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="desynth-workflow-")
        self.root = Path(self.directory.name)
        (self.root / "ai").mkdir()
        self.model = "ai/Qwen-Image-2512-Q4_K_M.gguf"
        self.adapter = "ai/Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors"
        for name in (self.model, self.adapter, "embeds_cache.pt"):
            (self.root / name).write_bytes(b"test fixture")
        self.patches = (
            patch.object(gui, "ROOT", self.root),
            patch.object(gui, "STORE", self.root / "out" / "gui"),
            patch.object(gui.Studio, "start_runtime_probe"),
        )
        for item in self.patches:
            item.start()
        self.studio = gui.Studio()
        self.studio.runtime = {
            "ready": True, "refreshing": False, "default_device": "cpu",
            "devices": [{"id": "cpu", "name": "CPU", "type": "cpu",
                         "precisions": ["float32"], "offload_modes": ["none"]}],
        }
        self.sample_calls = []
        self.sample_entered = threading.Event()
        self.sample_release = threading.Event()
        self.sample_release.set()
        self.fail_sampling = False
        self.studio.module = SimpleNamespace(
            torch=desynth.torch, build_pipeline=lambda *_args, **_kwargs: SimpleNamespace(num_timesteps=8),
            _sample=self.sample, _restore=desynth._restore,
            _free_pipeline=lambda: None, _is_oom=lambda _error: False,
        )
        self.studio.embeds = {
            "prompt_embeds": desynth.torch.zeros((2, 3)),
            "negative_prompt_embeds": desynth.torch.zeros((2, 3)),
            "prompt_embeds_mask": desynth.torch.ones(2, dtype=desynth.torch.int64),
            "negative_prompt_embeds_mask": desynth.torch.ones(2, dtype=desynth.torch.int64),
        }
        self.server = gui.LocalServer(("127.0.0.1", 0), QuietHandler)
        self.server.studio = self.studio
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.sample_release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.studio.close()
        for item in reversed(self.patches):
            item.stop()
        self.directory.cleanup()

    def request(self, path, payload=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            body = None if payload is None else json.dumps(payload).encode("utf-8")
            headers = {} if body is None else {"Content-Type": "application/json"}
            connection.request("GET" if body is None else "POST", path, body, headers)
            response = connection.getresponse()
            content = response.read()
            if "application/json" in response.getheader("Content-Type", ""):
                content = json.loads(content)
            return response.status, content
        finally:
            connection.close()

    def sample(self, pipe, image, _embeds, **params):
        self.sample_calls.append({**params, "size": image.size})
        self.sample_entered.set()
        if not self.sample_release.wait(3):
            raise RuntimeError("Sampling fixture was not released")
        if self.fail_sampling:
            raise RuntimeError("Synthetic inference failure")
        params["callback_on_step_end"](pipe, 3, 0, {})
        blurred = np.array(image.filter(ImageFilter.GaussianBlur(1.5)), dtype=np.int16)
        blurred[..., 0] += 7
        return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))

    def upload_source(self):
        # Uneven dimensions also exercise the model's multiple-of-16 working size.
        y, x = np.indices((49, 65))
        pixels = np.stack(((x * 17 + y * 3) % 256, (x * 5 + y * 19) % 256,
                           ((x // 4 + y // 3) % 2) * 255, (x * 13 + y * 7) % 256), axis=-1).astype(np.uint8)
        pixels[0, 0] = (251, 137, 53, 0)
        original = Image.fromarray(pixels, "RGBA")
        buffer = io.BytesIO()
        original.save(buffer, format="PNG")
        status, source = self.request("/api/upload", {
            "name": "transparent-source.png", "role": "source",
            "data": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
        })
        self.assertEqual(status, 200, source)
        return source, original

    def begin_job(self, source, **params):
        status, job = self.request("/api/process", {
            "source_id": source["id"], "params": {
                "transformer": self.model, "lightning": self.adapter, "device": "cpu",
                "offload": "none", "precision": "float32", "denoise": .25,
                "steps": 32, "seed": 42, **params,
            },
        })
        self.assertEqual(status, 202, job)
        return job

    def wait_for(self, predicate, message):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(.005)
        self.fail(message)

    def finish_job(self, job):
        def finished():
            status, current = self.request("/api/job/" + job["id"])
            self.assertEqual(status, 200, current)
            if current["status"] in ("done", "error") and self.studio.active is None:
                return current
            return None
        return self.wait_for(finished, "Processing job did not finish")

    def exported_image(self, asset):
        status, content = self.request(asset["url"])
        self.assertEqual(status, 200)
        with Image.open(io.BytesIO(content)) as png:
            self.assertEqual(png.format, "PNG")
            self.assertEqual(png.mode, "RGBA")
            image = png.copy()
        status, pixels = self.request("/api/pixels/" + asset["id"])
        self.assertEqual(status, 200)
        self.assertEqual(pixels, image.tobytes())
        return image

    def test_process_preview_comparison_and_png_preserve_the_complete_rgba_pair(self):
        source, original = self.upload_source()
        self.assertEqual(self.exported_image(source).tobytes(), original.tobytes())
        self.sample_release.clear()
        accepted = self.begin_job(source, passes=2, keep_intermediate=True)
        try:
            self.assertTrue(self.sample_entered.wait(2))
            status, session = self.request("/api/state")
            self.assertEqual(status, 200)
            self.assertEqual(session["active_job_id"], accepted["id"])
            status, conflict = self.request("/api/process", {"source_id": source["id"], "params": {}})
            self.assertEqual(status, 409, conflict)
        finally:
            self.sample_release.set()
        job = self.finish_job(accepted)
        self.assertEqual(job["status"], "done", job.get("error"))
        self.assertEqual(job["progress"], 100)
        self.assertEqual(job["preview_size"], [64, 48])
        self.assertEqual([call["seed"] for call in self.sample_calls], [42, 43])
        self.assertEqual([call["size"] for call in self.sample_calls], [(64, 48), (64, 48)])
        for call in self.sample_calls:
            self.assertEqual((call["denoise"], call["steps"], call["width"], call["height"]), (.25, 32, 64, 48))
        clean = self.exported_image(job["clean"])
        result = self.exported_image(job["result"])
        self.assertEqual(clean.size, (64, 48))
        self.assertEqual(clean.getchannel("A").getextrema(), (255, 255))
        self.assertEqual(result.size, original.size)
        self.assertEqual(result.getchannel("A").tobytes(), original.getchannel("A").tobytes())
        self.assertNotEqual(result.convert("RGB").tobytes(), original.convert("RGB").tobytes())
        status, comparison = self.request("/api/compare", {"source_id": source["id"], "result_id": job["result"]["id"]})
        self.assertEqual(status, 200, comparison)
        self.assertEqual(comparison, job["metrics"])
        self.assertEqual(comparison["mean_alpha"], 0)
        self.assertGreater(comparison["changed_pct"], 0)
        self.assertFalse(comparison["identical"])
        recorded = json.loads(result.info["desynth"])
        self.assertEqual(recorded["params"]["seed"], 42)
        self.assertEqual(recorded["transformer"], self.model)

        previews = []
        for revision, settings in enumerate(({"restore": False},
                                             {"restore": True, "restore_mode": "gaussian", "restore_sigma": .5},
                                             {"restore": True, "restore_mode": "edge", "restore_sigma": 2.5, "unsharp": .3}), 1):
            status, restored = self.request("/api/restore", {
                "source_id": source["id"], "clean_id": job["clean"]["id"],
                "client_id": "workflow", "revision": revision, "params": settings,
            })
            self.assertEqual(status, 200, restored)
            self.assertEqual(restored["revision"], revision)
            preview = self.exported_image(restored["result"])
            self.assertEqual(preview.size, original.size)
            self.assertEqual(preview.getchannel("A").tobytes(), original.getchannel("A").tobytes())
            self.assertEqual(json.loads(preview.info["desynth"])["params"]["seed"], 42)
            status, metrics = self.request("/api/compare", {
                "source_id": source["id"], "result_id": restored["result"]["id"],
            })
            self.assertEqual(status, 200, metrics)
            self.assertEqual(metrics, restored["metrics"])
            previews.append(preview)
        self.assertEqual(previews[0].convert("RGB").tobytes(),
                         clean.convert("RGB").resize(original.size, Image.Resampling.LANCZOS).tobytes())
        self.assertEqual(len({image.tobytes() for image in previews}), 3)
        self.assertEqual(len(self.sample_calls), 2, "Live restoration must not rerun inference")

    def test_latest_preview_revision_wins_over_running_and_pending_http_requests(self):
        source, original = self.upload_source()
        job = self.finish_job(self.begin_job(source, restore=False))
        self.assertEqual(job["status"], "done", job.get("error"))
        initial_results = {item["id"] for item in self.studio.assets.values() if item["kind"] == "result"}

        def request_restore(revision):
            return self.request("/api/restore", {
                "source_id": source["id"], "clean_id": job["clean"]["id"],
                "client_id": "preview-client", "revision": revision,
                "params": {"restore": True, "restore_mode": "edge", "restore_sigma": .5 + revision},
            })

        def pending(revision):
            with self.studio.lock:
                return self.studio.restore_pending.get("preview-client", {}).get("revision") == revision

        with ThreadPoolExecutor(max_workers=3) as executor:
            self.studio.cpu_lock.acquire()
            try:
                first = executor.submit(request_restore, 1)
                self.wait_for(lambda: self.studio.restore_running, "First preview did not start")
                second = executor.submit(request_restore, 2)
                self.wait_for(lambda: pending(2), "Second preview did not enter its pending slot")
                status, cancelled = self.request("/api/restore/cancel", {"client_id": "preview-client", "revision": 3})
                self.assertEqual((status, cancelled), (200, {"superseded": True}))
                self.assertEqual(second.result(timeout=2), (200, {"superseded": True}))
                latest = executor.submit(request_restore, 3)
                self.wait_for(lambda: pending(3), "Latest preview did not enter its pending slot")
            finally:
                self.studio.cpu_lock.release()
            self.assertEqual(first.result(timeout=2), (200, {"superseded": True}))
            status, restored = latest.result(timeout=3)
        self.assertEqual(status, 200, restored)
        self.assertEqual(restored["revision"], 3)
        image = self.exported_image(restored["result"])
        self.assertEqual(image.getchannel("A").tobytes(), original.getchannel("A").tobytes())
        new_results = {item["id"] for item in self.studio.assets.values() if item["kind"] == "result"} - initial_results
        self.assertEqual(new_results, {restored["result"]["id"]})
        status, obsolete = request_restore(2)
        self.assertEqual((status, obsolete), (200, {"superseded": True}))

    def test_failed_processing_is_observable_and_a_subsequent_job_can_finish(self):
        source, _original = self.upload_source()
        self.fail_sampling = True
        with patch.object(gui.traceback, "print_exc"):
            failed = self.finish_job(self.begin_job(source))
        self.assertEqual(failed["status"], "error")
        self.assertEqual(failed["error"], "Synthetic inference failure")
        self.assertIsNone(failed["result"])
        status, state = self.request("/api/state")
        self.assertEqual(status, 200)
        self.assertIsNone(state["active_job_id"])
        self.fail_sampling = False
        finished = self.finish_job(self.begin_job(source))
        self.assertEqual(finished["status"], "done", finished.get("error"))
        self.exported_image(finished["result"])


if __name__ == "__main__":
    unittest.main()
