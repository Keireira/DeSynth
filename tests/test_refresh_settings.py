"""Live model/runtime refresh regressions; no real CUDA or model loading."""
from __future__ import annotations

from contextlib import nullcontext
import http.client
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

import gui


class QuietHandler(gui.Handler):
    def log_message(self, *_args):
        pass


class RefreshTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="desynth-refresh-test-")
        self.root = Path(self.temporary.name)
        (self.root / "ai").mkdir()
        self.model = self.root / "ai" / "Q4.gguf"
        self.model.write_bytes(b"four")
        self.adapter = self.root / "ai" / "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors"
        self.adapter.write_bytes(b"adapter")
        (self.root / "embeds_cache.pt").write_bytes(b"embeddings")
        self.patches = (patch.object(gui, "ROOT", self.root), patch.object(gui, "STORE", self.root / "out" / "gui"))
        for item in self.patches:
            item.start()
        with patch.object(gui.Studio, "start_runtime_probe"):
            self.studio = gui.Studio()
        self.studio.runtime = {
            "ready": True, "refreshing": False, "default_device": "cpu",
            "devices": [{"id": "cpu", "name": "CPU", "type": "cpu", "precisions": ["float32"], "offload_modes": ["none"]}],
        }
        self.server = gui.LocalServer(("127.0.0.1", 0), QuietHandler)
        self.server.studio = self.studio
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()
        self.connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)

    def tearDown(self):
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def request(self, path, payload=None, headers=None):
        body = json.dumps(payload).encode() if payload is not None else None
        request_headers = {"Content-Type": "application/json"} if payload is not None else {}
        request_headers.update(headers or {})
        self.connection.request("POST" if payload is not None else "GET", path, body, request_headers)
        response = self.connection.getresponse()
        return response.status, json.loads(response.read())

    def test_refresh_discovers_added_and_moved_weights_without_restart(self):
        moved_model = self.root / "Q4.gguf"
        self.model.rename(moved_model)
        better_adapter = self.root / "ai" / "Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors"
        better_adapter.write_bytes(b"eight")
        self.adapter.unlink()
        (self.root / "ai" / "Q8.gguf").write_bytes(b"eight")
        with patch.object(self.studio, "start_runtime_probe") as probe:
            status, state = self.request("/api/refresh", {})
        self.assertEqual(status, 200, state)
        self.assertEqual([item["path"] for item in state["models"]], ["Q4.gguf", "ai/Q8.gguf"])
        self.assertEqual([item["path"] for item in state["lightning_adapters"]], ["ai/" + better_adapter.name])
        self.assertEqual(state["default_lightning"], "ai/" + better_adapter.name)
        self.assertEqual(state["default_steps"], 32)
        self.assertTrue(state["ready"], state["missing"])
        probe.assert_called_once_with()

    def test_refresh_reports_removed_models_and_keeps_active_job_and_assets(self):
        source = self.studio.save(Image.new("RGBA", (2, 2)), "source", "source.png")
        pipeline, embeddings = object(), object()
        self.studio.pipe, self.studio.embeds = pipeline, embeddings
        self.studio.active = "job-in-progress"
        self.studio.jobs["job-in-progress"] = {"status": "running", "runtime": {"device": "cpu"}}
        self.model.unlink()
        self.adapter.unlink()
        with patch.object(self.studio, "start_runtime_probe"):
            status, state = self.request("/api/refresh", {})
        self.assertEqual(status, 200)
        self.assertFalse(state["ready"])
        self.assertEqual(state["models"], [])
        self.assertEqual(state["lightning_adapters"], [])
        self.assertIsNone(state["default_lightning"])
        self.assertEqual(state["active_job_id"], "job-in-progress")
        self.assertEqual(state["assets"], [source])
        self.assertIs(self.studio.pipe, pipeline)
        self.assertIs(self.studio.embeds, embeddings)
        self.assertEqual(self.studio.jobs["job-in-progress"]["runtime"], {"device": "cpu"})

    def test_refresh_probes_once_and_keeps_existing_runtime_usable(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()

        def probe():
            entered.set()
            release.wait(2)

        original_probe = self.studio._probe_runtime

        def tracked_probe():
            try:
                original_probe()
            finally:
                finished.set()

        with patch.object(self.studio, "probe_device", side_effect=probe) as detect, patch.object(self.studio, "_probe_runtime", side_effect=tracked_probe):
            try:
                status, state = self.request("/api/refresh", {})
                self.assertEqual(status, 200)
                self.assertTrue(entered.wait(1))
                self.assertTrue(state["runtime"]["refreshing"])
                self.assertTrue(state["runtime"]["ready"])
                self.assertEqual(self.studio.resolve_runtime(gui.parameters({}, process=True))["device"], "cpu")
                self.request("/api/refresh", {})
                self.assertEqual(detect.call_count, 1)
            finally:
                release.set()
                self.assertTrue(finished.wait(2))
        status, state = self.request("/api/state")
        self.assertEqual(status, 200)
        self.assertFalse(state["runtime"]["refreshing"])
        self.assertFalse(self.studio.runtime_probe_running)

    def test_repeated_probe_replaces_stale_devices_instead_of_duplicating_them(self):
        fake_cuda = SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 1, device=lambda _index: nullcontext(),
            is_bf16_supported=lambda **_kwargs: True, get_device_name=lambda _index: "Replacement GPU",
        )
        fake_torch = SimpleNamespace(cuda=fake_cuda, backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
        self.studio.runtime["devices"].append({"id": "cuda:9", "name": "Removed GPU", "type": "cuda"})
        with patch.dict("sys.modules", {"torch": fake_torch}), patch.object(gui.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="Replacement GPU\n")):
            self.studio.probe_device()
            self.studio.probe_device()
            self.assertEqual([item["id"] for item in self.studio.runtime["devices"]], ["cpu", "cuda:0"])
            self.assertEqual(self.studio.runtime["devices"][1]["precisions"], ["bfloat16", "float32"])
            fake_cuda.is_available = lambda: False
            self.studio.probe_device()
        self.assertEqual([item["id"] for item in self.studio.runtime["devices"]], ["cpu"])
        self.assertEqual(self.studio.runtime["default_device"], "cpu")

    def test_accepted_job_pins_model_candidates_and_adapter_before_refresh(self):
        source = self.studio.save(Image.new("RGBA", (16, 16)), "source", "source.png")
        captured = {}
        started = threading.Event()

        def capture_job(_job_id, _source_id, params):
            captured.update(params)
            started.set()

        with patch.object(self.studio, "work", side_effect=capture_job), patch.object(self.studio, "start_runtime_probe"):
            status, job = self.request("/api/process", {"source_id": source["id"], "params": {}})
            self.assertEqual(status, 202, job)
            self.assertTrue(started.wait(1))
            (self.root / "ai" / "Q2.gguf").write_bytes(b"xx")
            (self.root / "ai" / "Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors").write_bytes(b"new adapter")
            self.request("/api/refresh", {})
        self.assertEqual(captured["_model_candidates"], ["ai/Q4.gguf"])
        self.assertEqual(captured["lightning"], "ai/" + self.adapter.name)
        self.assertEqual(job["runtime"], {"device": "cpu", "offload": "none", "precision": "float32", "device_name": "CPU"})

    def test_refresh_rejects_cross_origin_requests(self):
        with patch.object(self.studio, "refresh") as refresh:
            status, _error = self.request("/api/refresh", {}, {"Origin": "https://example.invalid"})
        self.assertEqual(status, 403)
        refresh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
