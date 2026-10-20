"""Local browser UI for DeSynth. Run: python gui.py [--port 8758]."""
from __future__ import annotations

import argparse
import base64
import binascii
import importlib.util
import io
import ipaddress
import json
import math
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image, UnidentifiedImageError
from PIL.PngImagePlugin import PngInfo
from model_catalog import (automatic_model_paths, default_steps_for_lightning,
                           discover_adapters, discover_models, effective_img2img_steps,
                           embeddings_path, lightning_steps)
from processing import finish_image, normalize_rgba, oriented_size, prepare_image, srgb_profile_bytes

ROOT = Path(__file__).resolve().parent
STORE = ROOT / "out" / "gui"
WEB = ROOT / "web"
MAX_BODY = 48 * 1024 * 1024
MAX_IMAGE = 32 * 1024 * 1024
MAX_PIXELS = 36_000_000
MAX_ASSETS = 128
MAX_TEMP_ASSETS = 48
MAX_TEMP_BYTES = 512 * 1024 * 1024
MAX_JOBS = 16
MAX_RESTORE_CLIENTS = 8
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def model_inventory(root):
    def item(path):
        return {"name": path.name, "path": path.relative_to(root).as_posix()}
    adapters = []
    for path in discover_adapters(root):
        steps = lightning_steps(path)
        precision = "FP32" if "fp32" in path.name.lower() else "BF16" if "bf16" in path.name.lower() else ""
        adapters.append({**item(path), "steps": steps,
                         "label": f"{steps} steps" + (f" · {precision}" if precision else "")})
    return [item(path) for path in discover_models(root)], adapters


def weight_identity(path):
    """Replacing a weight file at the same path must reload the pipeline."""
    info = path.stat()
    return str(path.resolve()), info.st_size, info.st_mtime_ns


class APIError(Exception):
    def __init__(self, message, status=400):
        self.status = status
        super().__init__(message)


def number(value, name, low, high, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise APIError(f"{name}: a finite number is required.")
    if not low <= value <= high or (integer and int(value) != value):
        raise APIError(f"{name}: expected a value between {low} and {high}.")
    return int(value) if integer else float(value)


def parameters(raw, process=False, default_steps=8):
    if not isinstance(raw, dict):
        raise APIError("params must be an object.")
    result = {
        "restore_sigma": number(raw.get("restore_sigma", 1.95), "restore_sigma", 0.1, 10),
        "unsharp": number(raw.get("unsharp", 0), "unsharp", 0, 3),
        "restore_mode": raw.get("restore_mode", "gaussian"),
        "restore": raw.get("restore", True),
    }
    if result["restore_mode"] not in ("gaussian", "edge"):
        raise APIError("restore_mode must be gaussian or edge.")
    if not isinstance(result["restore"], bool):
        raise APIError("restore must be a boolean.")
    if process:
        result.update(
            denoise=number(raw.get("denoise", .25), "denoise", .01, 1),
            steps=number(raw.get("steps", default_steps), "steps", 1, 64, True),
            passes=number(raw.get("passes", 1), "passes", 1, 8, True),
            seed=raw.get("seed"), transformer=raw.get("transformer"),
            preview=raw.get("preview", False), keep_intermediate=raw.get("keep_intermediate", False),
            device=raw.get("device", "auto"), offload=raw.get("offload", "auto"),
            precision=raw.get("precision", "auto"),
            lightning=raw.get("lightning"),
            lightning_scale=number(raw.get("lightning_scale", .8), "lightning_scale", .01, 2),
        )
        device = result["device"]
        if not isinstance(device, str) or not (device in ("auto", "cpu", "mps") or device.startswith("cuda:") and device[5:].isdigit()):
            raise APIError("Unknown device. Choose auto, cpu, mps or an available cuda:N device.")
        if result["offload"] not in ("auto", "sequential", "model", "none"):
            raise APIError("offload must be auto, sequential, model or none.")
        if result["precision"] not in ("auto", "bfloat16", "float16", "float32"):
            raise APIError("precision must be auto, bfloat16, float16 or float32.")
        if result["seed"] is not None:
            result["seed"] = number(result["seed"], "seed", 0, 2**63 - 9, True)
        if effective_img2img_steps(result["steps"], result["denoise"]) < 1:
            raise APIError("These settings produce no processing steps. Increase steps or denoise.")
        for field in ("preview", "keep_intermediate"):
            if not isinstance(result[field], bool):
                raise APIError(f"{field} must be a boolean.")
    return result


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


class Studio:
    def __init__(self):
        STORE.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.cpu_lock = threading.Lock()
        self.assets, self.paths, self.jobs, self.clean_sources, self.started = {}, {}, {}, {}, {}
        self.temporary_assets = {}
        self.restore_clients = OrderedDict()
        self.restore_pending = OrderedDict()
        self.restore_running = self.restore_worker = None
        self.closed = False
        self.latest_source = self.latest_job = None
        self.active = None
        self.pipe = self.pipe_path = self.pipe_key = self.pipe_scale = self.embeds = self.module = None
        self.device = "Detecting device…"
        self.runtime = {"ready": False, "refreshing": False, "devices": [{"id": "cpu", "name": "CPU", "type": "cpu",
                         "precisions": ["float32"], "offload_modes": ["none"]}], "default_device": "cpu"}
        self.runtime_probe_running = False
        self.models, self.lightning_adapters = model_inventory(ROOT)
        self.default_lightning = self.lightning_adapters[0]["path"] if self.lightning_adapters else None
        self.default_steps = default_steps_for_lightning(self.default_lightning or "")
        self.default_source = self.default_result = None
        originals = sorted(p for p in ROOT.iterdir() if p.name.lower() == "original.png")
        if originals:
            self.default_source = self.register(originals[0], "source")["id"]
        self.discover_results()
        self.start_runtime_probe()

    def discover_results(self):
        """Read recorded outputs; legacy discovery accepts only matching images."""
        if not self.default_source:
            return
        original = self.assets[self.default_source]
        source_path = self.paths[self.default_source].resolve()
        output_root = (ROOT / "out").resolve()
        candidates = set()
        for manifest in output_root.glob("*_desynth_*.json"):
            try:
                record = json.loads(manifest.read_text(encoding="utf-8"))
                if Path(record["source"]["path"]).resolve() != source_path:
                    continue
                for output in record.get("outputs", []):
                    path = Path(output["path"]).resolve()
                    if output.get("kind") == "result" and path.is_relative_to(output_root) and path.is_file():
                        candidates.add(path)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        for path in output_root.iterdir():
            if (path.is_file() and path.name.casefold().startswith("original_desynth")
                    and path.suffix.casefold() == ".png"
                    and not re.search(r"(?:^|_)(?:diff|comparison|clean)(?:_|\.|$)", path.name, re.IGNORECASE)):
                candidates.add(path.resolve())
        valid = []
        for path in candidates:
            try:
                with Image.open(path) as image:
                    if oriented_size(image) == (original["width"], original["height"]):
                        valid.append((path.stat().st_mtime_ns, path))
            except (OSError, ValueError, Image.DecompressionBombError):
                continue
        for _, path in sorted(valid)[-MAX_JOBS:]:
            self.default_result = self.register(path, "result")["id"]

    def start_runtime_probe(self):
        """Refresh capabilities without unloading the pipeline or changing a job."""
        with self.lock:
            if self.runtime_probe_running:
                return
            self.runtime_probe_running = True
            self.runtime = {**self.runtime, "refreshing": True}
        threading.Thread(target=self._probe_runtime, daemon=True).start()

    def _probe_runtime(self):
        try:
            self.probe_device()
        finally:
            with self.lock:
                self.runtime = {**self.runtime, "refreshing": False}
                self.runtime_probe_running = False

    def refresh(self):
        models, adapters = model_inventory(ROOT)
        with self.lock:
            self.models, self.lightning_adapters = models, adapters
            self.default_lightning = adapters[0]["path"] if adapters else None
            self.default_steps = default_steps_for_lightning(self.default_lightning or "")
        self.start_runtime_probe()
        return self.state()

    def probe_device(self):
        devices = [{"id": "cpu", "name": "CPU", "type": "cpu", "precisions": ["float32"], "offload_modes": ["none"]}]
        try:
            probe = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
            pending = f"{probe.stdout.strip().splitlines()[0]} · checking CUDA" if probe.returncode == 0 and probe.stdout.strip() else "Checking device…"
        except (OSError, subprocess.TimeoutExpired):
            pending = "Checking device…"
        with self.lock:
            if not self.runtime["ready"]:
                self.device = pending
        try:
            import torch
            if torch.cuda.is_available():
                for index in range(torch.cuda.device_count()):
                    with torch.cuda.device(index):
                        bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
                    devices.append({"id": f"cuda:{index}", "name": torch.cuda.get_device_name(index), "type": "cuda",
                                    "precisions": (["bfloat16"] if bf16 else []) + ["float32"],
                                    "offload_modes": ["sequential", "model", "none"]})
            if torch.backends.mps.is_available():
                precisions = ["float32"]
                try:
                    probe_tensor = torch.zeros(1, device="mps", dtype=torch.bfloat16)
                    del probe_tensor
                    precisions.insert(0, "bfloat16")
                except (RuntimeError, TypeError):
                    pass
                devices.append({"id": "mps", "name": "Apple Silicon", "type": "mps", "precisions": precisions,
                                "offload_modes": ["model", "none"]})
            default = next((d for d in devices if d["type"] == "cuda"), next((d for d in devices if d["type"] == "mps"), devices[0]))
            with self.lock:
                self.runtime = {"ready": True, "refreshing": self.runtime.get("refreshing", False),
                                "devices": devices, "default_device": default["id"]}
                self.device = f"{default['type'].upper()} · {default['name']}" if default["type"] != "cpu" else "CPU"
        except Exception as exc:
            with self.lock:
                self.runtime = {"ready": True, "refreshing": self.runtime.get("refreshing", False),
                                "devices": devices[:1], "default_device": "cpu", "error": str(exc)}
                self.device = "CPU"

    def resolve_runtime(self, params):
        with self.lock:
            inventory = self.runtime
        if not inventory["ready"]:
            raise APIError("Devices are still being checked. Please retry in a moment.")
        device_id = inventory["default_device"] if params["device"] == "auto" else params["device"]
        capability = next((device for device in inventory["devices"] if device["id"] == device_id), None)
        if capability is None:
            raise APIError(f"Device {device_id} is unavailable. Select a device from the available list.")
        offload = params["offload"]
        if offload == "auto":
            offload = "sequential" if capability["type"] == "cuda" else "model" if capability["type"] == "mps" else "none"
        if offload not in capability["offload_modes"]:
            raise APIError(f"{offload} offload is not supported on {capability['name']}. Choose {', '.join(capability['offload_modes'])}.")
        precision = params["precision"]
        if precision == "auto":
            precision = "bfloat16" if "bfloat16" in capability["precisions"] else "float32"
        if precision not in capability["precisions"]:
            if precision == "float16":
                raise APIError("Qwen-Image can overflow in float16 and produce a black image. Choose bfloat16 (if available) or float32.")
            raise APIError(f"{precision} precision is not supported on {capability['name']}. Choose {', '.join(capability['precisions'])}.")
        return {"device": device_id, "offload": offload, "precision": precision, "device_name": capability["name"]}

    def register(self, path, kind, name=None, *, temporary=False):
        with Image.open(path) as image:
            width, height = oriented_size(image)
            if width * height > MAX_PIXELS:
                raise ValueError("Image is too large. Maximum: 36 megapixels.")
        asset_id = secrets.token_hex(16)
        asset = dict(id=asset_id, name=name or path.name, width=width, height=height,
                     url=f"/api/image/{asset_id}", kind=kind)
        with self.lock:
            self.assets[asset_id], self.paths[asset_id] = asset, path
            if temporary:
                self.temporary_assets[asset_id] = path.stat().st_size
            self.prune_assets(keep={asset_id})
        return asset

    def asset(self, asset_id, kind=None):
        with self.lock:
            if not isinstance(asset_id, str) or asset_id not in self.assets:
                raise APIError("Image not found. Please upload it again.", 404)
            asset = self.assets[asset_id]
            if kind and asset["kind"] != kind:
                raise APIError(f"Expected an image of type {kind}.")
            return asset

    def image(self, asset_id):
        with self.lock:
            self.asset(asset_id)
            with Image.open(self.paths[asset_id]) as image:
                return normalize_rgba(image)

    def image_bytes(self, asset_id):
        with self.lock:
            self.asset(asset_id)
            path = self.paths[asset_id]
            with Image.open(path) as image:
                profile = image.info.get("icc_profile")
                if (image.getexif().get(274) in (2, 3, 4, 5, 6, 7, 8)
                        or profile is not None and profile != srgb_profile_bytes()):
                    # Registered originals may predate upload normalization.
                    # Display exactly the same sRGB pixels used by metrics/diff.
                    data = io.BytesIO()
                    normalize_rgba(image).save(data, format="PNG")
                    return data.getvalue()
            return path.read_bytes()

    def save(self, image, kind, name, *, temporary=False, metadata=None):
        directory = STORE / "temporary" if temporary else STORE
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{secrets.token_hex(16)}.png"
        pnginfo = None
        if metadata is not None:
            pnginfo = PngInfo()
            pnginfo.add_text("desynth", json.dumps(metadata, ensure_ascii=True))
        image.save(path, format="PNG", pnginfo=pnginfo)
        try:
            return self.register(path, kind, name, temporary=temporary)
        except Exception:
            # This is the exact file created above, never an imported output.
            path.unlink(missing_ok=True)
            raise

    def forget_asset(self, asset_id):
        """Evict metadata; only files explicitly owned as temporary are removed."""
        path = self.paths.get(asset_id)
        if asset_id in self.temporary_assets and path is not None:
            if path.resolve().parent != (STORE / "temporary").resolve():
                raise RuntimeError("Refusing to remove a temporary asset outside its directory.")
            path.unlink(missing_ok=True)
        self.temporary_assets.pop(asset_id, None)
        self.assets.pop(asset_id, None)
        self.paths.pop(asset_id, None)
        self.clean_sources.pop(asset_id, None)

    def prune_assets(self, keep=()):
        with self.lock:
            protected = set(keep) | {self.default_source, self.default_result, self.latest_source}
            for job_id in (self.active, self.latest_job):
                job = self.jobs.get(job_id, {})
                protected.add(job.get("source_id"))
                for role in ("clean", "result"):
                    protected.add((job.get(role) or {}).get("id"))
            for client in self.restore_clients.values():
                protected.update(client.get("assets", ()))
                protected.update(client.get("results", ()))
            for ticket in [self.restore_running, *self.restore_pending.values()]:
                if ticket:
                    protected.update((ticket["source_id"], ticket["clean_id"]))
            for asset_id in list(self.assets):
                count_over = len(self.assets) > MAX_ASSETS
                temp_over = (len(self.temporary_assets) > MAX_TEMP_ASSETS
                             or sum(self.temporary_assets.values()) > MAX_TEMP_BYTES)
                if asset_id not in protected and (count_over or temp_over and asset_id in self.temporary_assets):
                    self.forget_asset(asset_id)

    def close(self):
        with self.lock:
            self.closed = True
            for ticket in self.restore_pending.values():
                self.finish_restore(ticket, {"superseded": True})
            self.restore_pending.clear()
            worker = self.restore_worker
        if worker:
            worker.join(timeout=30)
        with self.lock:
            for asset_id in list(self.temporary_assets):
                self.forget_asset(asset_id)

    def state(self):
        with self.lock:
            missing = [] if embeddings_path(ROOT).is_file() else ["embeds_cache.pt"]
            if not self.models:
                missing.append("GGUF transformer")
            if not self.lightning_adapters:
                missing.append("Lightning adapter")
            missing += [name for name in ("torch", "diffusers", "numpy", "cv2", "skimage") if importlib.util.find_spec(name) is None]
            return dict(device=self.device, ready=not missing, models=self.models, lightning_adapters=self.lightning_adapters,
                        default_lightning=self.default_lightning, default_steps=self.default_steps, assets=list(self.assets.values()),
                        default_source_id=self.default_source, default_result_id=self.default_result,
                        missing=missing, active_job_id=self.active, runtime=self.runtime)

    def upload(self, payload):
        role = payload.get("role", "source")
        if role not in ("source", "result"):
            raise APIError("role must be source or result.")
        data = payload.get("data")
        if not isinstance(data, str):
            raise APIError("Image data is missing.")
        if data.startswith("data:"):
            if ";base64," not in data[:128]:
                raise APIError("A base64-encoded image is required.")
            data = data.split(",", 1)[1]
        try:
            decoded = base64.b64decode(data, validate=True)
            if len(decoded) > MAX_IMAGE:
                raise APIError("File is too large. Maximum: 32 MB.", 413)
            with Image.open(io.BytesIO(decoded)) as source:
                if source.width * source.height > MAX_PIXELS:
                    raise APIError("Image is too large. Maximum: 36 megapixels.", 413)
                image = normalize_rgba(source)
        except ValueError as exc:
            raise APIError(str(exc)) from exc
        except (binascii.Error, OSError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise APIError("Cannot read this image. Use PNG, JPEG or WebP up to 36 megapixels.") from exc
        name = payload.get("name", "image.png")
        if not isinstance(name, str):
            raise APIError("name must be a string.")
        asset = self.save(image, role, Path(name.replace("\\", "/")).name[:160] or "image.png", temporary=True)
        with self.lock:
            if role == "source":
                self.latest_source = asset["id"]
            self.prune_assets()
        return asset

    def compare(self, source_id, result_id):
        return self.compare_images(self.image(source_id), self.image(result_id))

    @staticmethod
    def compare_images(a, b):
        import numpy as np
        from compare import basic_metrics
        if a.size != b.size:
            raise APIError("Image dimensions differ. Comparison requires images of the same size.")
        aa, bb = np.asarray(a), np.asarray(b)
        result = basic_metrics(aa[..., :3], bb[..., :3])
        changed, alpha = 0, 0.0
        for row in range(0, a.height, 128):
            left, right = aa[row:row + 128], bb[row:row + 128]
            changed += int(np.count_nonzero(np.any(left != right, axis=2)))
            alpha += float(np.abs(left[..., 3].astype("float32") - right[..., 3]).sum(dtype=np.float64))
        pixels = a.width * a.height
        result.update(changed_pct=changed / pixels * 100, identical=changed == 0, mean_alpha=alpha / pixels)
        return finite_json(result)

    def restore_image(self, clean, original, params):
        return finish_image(clean, original, params, getattr(self.module, "_restore", None))

    def quick_restore(self, payload):
        source_id, clean_id = payload.get("source_id"), payload.get("clean_id")
        params = parameters(payload.get("params", {}))
        with self.lock:
            self.asset(source_id, "source")
            self.asset(clean_id, "clean")
            if self.clean_sources.get(clean_id) != source_id:
                raise APIError("This clean result belongs to a different source image.")
            client_id, client, revision = self.restore_client(payload)
            if revision < client["revision"]:
                return {"superseded": True}
            for ticket in (self.restore_running, self.restore_pending.get(client_id)):
                if ticket and ticket["client_id"] == client_id and ticket["revision"] == revision:
                    return {"superseded": True}
            self.supersede_restore(client_id, revision)
            ticket = dict(client_id=client_id, revision=revision, source_id=source_id,
                          clean_id=clean_id, params=params, event=threading.Event())
            client["assets"] = (source_id, clean_id)
            self.restore_pending[client_id] = ticket
            if self.restore_worker is None:
                self.restore_worker = threading.Thread(target=self.restore_work, daemon=True)
                self.restore_worker.start()
        # Replaced pending requests wake immediately, so handlers do not build
        # an unbounded queue while the CPU is processing a previous image.
        if not ticket["event"].wait(60):
            with self.lock:
                ticket["cancelled"] = True
                if self.restore_pending.get(client_id) is ticket:
                    self.restore_pending.pop(client_id)
            raise APIError("Restoration timed out. Please try again.", 503)
        if "error" in ticket:
            raise ticket["error"]
        return ticket["response"]

    def restore_client(self, payload):
        if self.closed:
            raise APIError("The server is stopping.", 503)
        client_id = payload.get("client_id", "legacy")
        if not isinstance(client_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", client_id):
            raise APIError("client_id must contain 1–80 letters, digits, hyphens or underscores.")
        client = self.restore_clients.get(client_id)
        revision = payload.get("revision", (client or {}).get("revision", 0) + 1)
        revision = number(revision, "revision", 0, 2**53 - 1, True)
        if client is None:
            if len(self.restore_clients) >= MAX_RESTORE_CLIENTS:
                evict = next((key for key in self.restore_clients if key not in self.restore_pending
                              and (not self.restore_running or self.restore_running["client_id"] != key)), None)
                if evict is None:
                    raise APIError("Too many simultaneous preview sessions. Retry shortly.", 429)
                self.restore_clients.pop(evict)
            client = {"revision": -1, "assets": (), "results": []}
            self.restore_clients[client_id] = client
        self.restore_clients.move_to_end(client_id)
        return client_id, client, revision

    def supersede_restore(self, client_id, revision):
        client = self.restore_clients[client_id]
        if revision < client["revision"]:
            return
        client["revision"] = revision
        pending = self.restore_pending.get(client_id)
        if pending and pending["revision"] < revision:
            self.restore_pending.pop(client_id)
            self.finish_restore(pending, {"superseded": True})

    def cancel_restore(self, payload):
        with self.lock:
            client_id, client, revision = self.restore_client(payload)
            self.supersede_restore(client_id, revision)
            self.prune_assets()
            return {"superseded": True}

    def restore_obsolete(self, ticket):
        with self.lock:
            client = self.restore_clients.get(ticket["client_id"])
            return (self.closed or ticket.get("cancelled", False) or client is None
                    or client["revision"] != ticket["revision"])

    @staticmethod
    def finish_restore(ticket, response=None, error=None):
        if error is not None:
            ticket["error"] = error
        else:
            ticket["response"] = response
        ticket["event"].set()

    def restore_work(self):
        while True:
            with self.lock:
                if not self.restore_pending or self.closed:
                    self.restore_worker = None
                    self.restore_running = None
                    return
                _, ticket = self.restore_pending.popitem(last=False)
                self.restore_running = ticket
            try:
                response = {"superseded": True}
                with self.cpu_lock:
                    if not self.restore_obsolete(ticket):
                        original, clean = self.image(ticket["source_id"]), self.image(ticket["clean_id"])
                        if not self.restore_obsolete(ticket):
                            image = self.restore_image(clean, original, ticket["params"])
                            if not self.restore_obsolete(ticket):
                                metrics = self.compare_images(original, image)
                                if not self.restore_obsolete(ticket):
                                    try:
                                        metadata = json.loads(clean.info.get("desynth", "{}"))
                                    except (ValueError, TypeError):
                                        metadata = {}
                                    if not isinstance(metadata, dict):
                                        metadata = {}
                                    recorded = metadata.get("params", {})
                                    metadata["params"] = {**(recorded if isinstance(recorded, dict) else {}), **ticket["params"]}
                                    asset = self.save(image, "result", "desynth_restored.png", temporary=True,
                                                      metadata=metadata)
                                    with self.lock:
                                        if self.restore_obsolete(ticket):
                                            self.forget_asset(asset["id"])
                                        else:
                                            client = self.restore_clients[ticket["client_id"]]
                                            client["results"] = [*client["results"][-1:], asset["id"]]
                                            self.prune_assets()
                                            response = dict(result=asset, metrics=metrics, revision=ticket["revision"])
                self.finish_restore(ticket, response)
            except Exception as exc:
                self.finish_restore(ticket, error=exc)
            finally:
                with self.lock:
                    self.restore_running = None

    def process(self, payload):
        source_id = payload.get("source_id")
        self.asset(source_id, "source")
        with self.lock:
            if self.active:
                raise APIError("Processing is already running. Wait for the current job to finish.", 409)
            raw_params = payload.get("params", {})
            preferred = next((item for item in self.lightning_adapters if isinstance(raw_params, dict) and item["path"] == raw_params.get("lightning")), None)
            default_steps = default_steps_for_lightning(preferred["path"]) if preferred else self.default_steps
            params = parameters(raw_params, process=True, default_steps=default_steps)
            runtime = self.resolve_runtime(params)
            requested = params["transformer"]
            if requested is not None and requested not in [model["path"] for model in self.models]:
                raise APIError("Select a GGUF model from the available models. Use Refresh in Settings to update the list.")
            adapter = params["lightning"] if params["lightning"] is not None else self.default_lightning
            selected = next((item for item in self.lightning_adapters if item["path"] == adapter), None)
            if selected is None:
                raise APIError("Select a Lightning adapter from the available adapters. Use Refresh in Settings to update the list.")
            if not (ROOT / selected["path"]).is_file():
                raise APIError("The selected Lightning adapter was moved or removed. Use Refresh in Settings to update the list.")
            params["lightning"] = selected["path"]
            candidates = [requested] if requested else automatic_model_paths(self.models, ROOT)
            if not candidates or any(not (ROOT / path).is_file() for path in candidates):
                raise APIError("The selected GGUF model was moved or removed. Use Refresh in Settings to update the list.")
            # A concurrent refresh must not change an accepted job's fallback order.
            params["_model_candidates"] = candidates
            if not self.state()["ready"]:
                raise APIError("Environment is not ready. Run make setup; check GGUF and Lightning weights in ai/ and embeds_cache.pt.", 503)
            if params["seed"] is None:
                params["seed"] = secrets.randbits(31)
            job_id = secrets.token_hex(16)
            job = dict(id=job_id, status="queued", progress=0, stage="Queued", elapsed=0,
                       seed=params["seed"], source_id=source_id, result=None, clean=None, metrics=None, error=None)
            job.update(runtime=runtime, requested_runtime={key: params[key] for key in ("device", "offload", "precision")})
            job.update(lightning=params["lightning"], lightning_scale=params["lightning_scale"], steps=params["steps"])
            job.update(denoise=params["denoise"], starting_sigma=params["denoise"], sampling="linear-sigma-v1")
            self.jobs[job_id] = job
            self.started[job_id] = time.monotonic()
            self.active = job_id
            while len(self.jobs) > MAX_JOBS:
                oldest = next(key for key in self.jobs if key not in (self.active, self.latest_job))
                self.jobs.pop(oldest)
                self.started.pop(oldest, None)
            self.prune_assets()
        threading.Thread(target=self.work, args=(job_id, source_id, params), daemon=True).start()
        return dict(job)

    def work(self, job_id, source_id, params):
        job, started = self.jobs[job_id], self.started[job_id]
        failed = False

        def update(**values):
            with self.lock:
                job.update(values, elapsed=round(time.monotonic() - started, 1))

        try:
            update(status="running", stage="Loading libraries", progress=2)
            if self.module is None:
                import desynth
                self.module = desynth
            ds = self.module
            runtime = job["runtime"]
            dtype = getattr(ds.torch, runtime["precision"])
            original = self.image(source_id)
            image = prepare_image(original, params["preview"])
            update(preview_size=list(image.size), stage="Loading embeddings", progress=5)
            if self.embeds is None:
                self.embeds = ds.torch.load(embeddings_path(ROOT), map_location="cpu", weights_only=False)
            # Cached embeddings stay on CPU. Each job gets the selected precision
            # and device, including masks needed when no offload hooks are active.
            embeds = {}
            for key in ("prompt_embeds", "negative_prompt_embeds", "prompt_embeds_mask", "negative_prompt_embeds_mask"):
                value = self.embeds[key]
                embeds[key] = value.to(device=runtime["device"], dtype=dtype if value.is_floating_point() else value.dtype)
            candidates = params["_model_candidates"]
            clean = None
            for candidate in candidates:
                try:
                    pipe_key = (weight_identity(ROOT / candidate), weight_identity(ROOT / params["lightning"]),
                                runtime["device"], runtime["offload"], runtime["precision"])
                    if self.pipe_key != pipe_key or self.pipe is None:
                        self.drop_pipeline()
                        update(stage=f"Loading {candidate} (first run takes longer)", progress=7)
                        self.pipe = ds.build_pipeline(ROOT / candidate, device=runtime["device"],
                                                      offload=runtime["offload"], compute_dtype=dtype,
                                                      lightning_lora=ROOT / params["lightning"], lightning_scale=params["lightning_scale"])
                        self.pipe_path = candidate
                        self.pipe_key = pipe_key
                        self.pipe_scale = params["lightning_scale"]
                    elif self.pipe_scale != params["lightning_scale"]:
                        self.pipe.set_adapters(["lightning"], adapter_weights=[params["lightning_scale"]])
                        self.pipe_scale = params["lightning_scale"]
                    current = image
                    for pass_index in range(params["passes"]):
                        def step_end(pipe, step, timestep, callback_kwargs):
                            count = max(1, getattr(pipe, "num_timesteps", int(params["steps"] * params["denoise"])))
                            update(stage=f"Pass {pass_index + 1}/{params['passes']} · step {step + 1}/{count}",
                                   progress=min(88, 12 + 76 * (pass_index + (step + 1) / count) / params["passes"]))
                            return callback_kwargs
                        update(stage=f"Pass {pass_index + 1}/{params['passes']} · encoding image")
                        with ds.torch.inference_mode():
                            current = ds._sample(self.pipe, current, embeds, denoise=params["denoise"],
                                                 steps=params["steps"], seed=params["seed"] + pass_index,
                                                 width=image.width, height=image.height,
                                                 callback_on_step_end=step_end)
                    clean = current
                    break
                except Exception as exc:
                    if ds._is_oom(exc) and candidate != candidates[-1]:
                        update(stage="Out of memory: trying a smaller model", progress=7)
                    else:
                        raise
                # The failed call's traceback can own tensors even after
                # self.pipe is cleared. Leave its exception scope first.
                self.drop_pipeline(force=True)
            update(stage="Restoring details", progress=90)
            metadata = dict(
                sampling="linear-sigma-v1", starting_sigma=params["denoise"],
                params={key: value for key, value in params.items() if not key.startswith("_")},
                transformer=self.pipe_path, runtime=runtime, working_size=list(image.size),
                source_name=self.assets[source_id]["name"],
            )
            clean_asset = self.save(clean.convert("RGBA"), "clean", "desynth_clean.png",
                                    temporary=not params["keep_intermediate"], metadata=metadata)
            with self.lock:
                self.clean_sources[clean_asset["id"]] = source_id
                update(clean=clean_asset)
            with self.cpu_lock:
                final = self.restore_image(clean, original, params)
                update(stage="Comparing RGBA channels", progress=95)
                metrics = self.compare_images(original, final)
            result = self.save(final, "result", f"{Path(self.assets[source_id]['name']).stem}_desynth_seed{params['seed']}.png",
                               metadata=metadata)
            update(result=result)
            update(status="done", stage="Done", progress=100, metrics=metrics)
            with self.lock:
                self.latest_job = job_id
        except Exception as exc:
            failed = True
            traceback.print_exc()
            message = str(exc) or type(exc).__name__
            if self.module and self.module._is_oom(exc):
                message = "Out of memory. Use a smaller image, choose sequential offload on CUDA, or select a Q2 GGUF model. " + message
            update(status="error", stage="Processing failed", error=message[:1800])
        finally:
            try:
                if failed:
                    self.drop_pipeline(force=True)
            finally:
                with self.lock:
                    self.active = None
                    self.prune_assets()

    def drop_pipeline(self, *, force=False):
        if self.pipe is not None or force:
            self.pipe, self.pipe_path, self.pipe_key, self.pipe_scale = None, None, None, None
            if self.module is not None:
                self.module._free_pipeline()


class LocalServer(ThreadingHTTPServer):
    # Browser image and API requests arrive in bursts, especially during startup.
    request_queue_size = 64


class Handler(BaseHTTPRequestHandler):
    server_version = "DeSynth/1.0"
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True
    timeout = 30

    @property
    def studio(self):
        return self.server.studio

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            # Browsers may abort a pending restore or close an idle connection.
            self.close_connection = True

    def send_json(self, status, value):
        data = json.dumps(finite_json(value), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_bytes(status, data, "application/json; charset=utf-8")

    def log_request(self, code="-", size="-"):
        if str(code) == "200" and urlsplit(self.path).path.startswith("/api/job/"):
            return
        super().log_request(code, size)

    def send_bytes(self, status, data, content_type, filename=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Connection", "close" if self.close_connection else "keep-alive")
        if filename:
            self.send_header("Content-Disposition", 'inline; filename="image.png"')
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def check_origin(self, mutate=False):
        host = self.headers.get("Host", "")
        parsed = urlsplit("http://" + host)
        try:
            valid_port = parsed.port == self.server.server_port
        except ValueError:
            valid_port = False
        if parsed.hostname not in ("localhost", "127.0.0.1", "::1") or not valid_port or parsed.username or parsed.password:
            raise APIError("Only the local server address is allowed.", 403)
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + host:
            raise APIError("Requests from another origin are not allowed.", 403)
        if mutate and self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise APIError("Requests from another origin are not allowed.", 403)

    def do_GET(self):
        try:
            self.check_origin()
            path = urlsplit(self.path).path
            if path == "/health":
                return self.send_json(200, {"ok": True})
            if path == "/favicon.ico":
                return self.send_bytes(204, b"", "image/x-icon")
            if path == "/api/state":
                return self.send_json(200, self.studio.state())
            if path.startswith("/api/image/"):
                asset_id = path.removeprefix("/api/image/")
                return self.send_bytes(200, self.studio.image_bytes(asset_id), "image/png")
            if path.startswith("/api/pixels/"):
                asset_id = path.removeprefix("/api/pixels/")
                return self.send_bytes(200, self.studio.image(asset_id).tobytes(), "application/octet-stream")
            if path.startswith("/api/job/"):
                with self.studio.lock:
                    job = self.studio.jobs.get(path.removeprefix("/api/job/"))
                    if not job:
                        raise APIError("Job not found.", 404)
                    if job["status"] in ("running", "queued"):
                        job["elapsed"] = round(time.monotonic() - self.studio.started[job["id"]], 1)
                    return self.send_json(200, dict(job))
            files = {"/": ("index.html", "text/html; charset=utf-8"), "/index.html": ("index.html", "text/html; charset=utf-8"),
                     "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                     "/diff-worker.js": ("diff-worker.js", "text/javascript; charset=utf-8"),
                     "/style.css": ("style.css", "text/css; charset=utf-8"), "/styles.css": ("style.css", "text/css; charset=utf-8")}
            if path not in files:
                raise APIError("Not found.", 404)
            filename, mime = files[path]
            return self.send_bytes(200, (WEB / filename).read_bytes(), mime)
        except APIError as exc:
            self.send_json(exc.status, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def do_POST(self):
        # Remember what remains unread even when Origin/Content-Type rejects a
        # request before JSON parsing. Closing a Windows socket with unread data
        # can reset the connection before the error response reaches the client.
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = None
        remaining = length if length is not None and 0 <= length <= MAX_BODY and not self.headers.get("Transfer-Encoding") else None
        try:
            self.check_origin(mutate=True)
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
                raise APIError("Content-Type: application/json is required.", 415)
            if self.headers.get("Transfer-Encoding"):
                raise APIError("Transfer-Encoding is not supported. Send Content-Length.")
            if length is None:
                raise APIError("Invalid request length.")
            if not 0 < length <= MAX_BODY:
                raise APIError("Request is too large or empty. Maximum: 48 MB.", 413)
            body = self.rfile.read(length)
            remaining -= len(body)
            if remaining:
                raise APIError("Incomplete request body.")
            try:
                payload = json.loads(body)
            except (ValueError, UnicodeError) as exc:
                raise APIError("Invalid JSON.") from exc
            if not isinstance(payload, dict):
                raise APIError("A JSON object is required.")
            path = urlsplit(self.path).path
            if path == "/api/refresh":
                return self.send_json(200, self.studio.refresh())
            if path == "/api/upload":
                return self.send_json(200, self.studio.upload(payload))
            if path == "/api/process":
                return self.send_json(202, self.studio.process(payload))
            if path == "/api/restore":
                return self.send_json(200, self.studio.quick_restore(payload))
            if path == "/api/restore/cancel":
                return self.send_json(200, self.studio.cancel_restore(payload))
            if path == "/api/compare":
                with self.studio.cpu_lock:
                    return self.send_json(200, self.studio.compare(payload.get("source_id"), payload.get("result_id")))
            raise APIError("Not found.", 404)
        except APIError as exc:
            if remaining:
                previous_timeout = self.connection.gettimeout()
                self.connection.settimeout(1)
                try:
                    while remaining:
                        chunk = self.rfile.read(min(64 * 1024, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                except (OSError, TimeoutError):
                    pass
                finally:
                    self.connection.settimeout(previous_timeout)
            # Fully consumed requests can keep their connection, including
            # normal parameter errors. Unbounded/incomplete bodies cannot.
            self.close_connection = self.close_connection or remaining != 0
            self.send_json(exc.status, {"error": str(exc)})
            if remaining != 0:
                try:
                    self.connection.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            traceback.print_exc()
            self.send_json(500, {"error": str(exc)})


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="DeSynth local GUI")
    parser.add_argument("--host", default="127.0.0.1", help="loopback address only (127.0.0.1, localhost, ::1)")
    parser.add_argument("--port", type=int, default=8758)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    try:
        local = args.host == "localhost" or ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        local = False
    if not local or args.host not in ("localhost", "127.0.0.1", "::1"):
        parser.error("--host must be 127.0.0.1, localhost or ::1. This GUI only runs locally.")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535.")
    if args.host == "::1":
        LocalServer.address_family = socket.AF_INET6
    studio = Studio()
    try:
        server = LocalServer((args.host, args.port), Handler)
    except OSError as exc:
        parser.exit(1, f"Cannot start the server: {exc}. Choose another --port.\n")
    server.studio = studio
    url = f"http://{'[::1]' if args.host == '::1' else args.host}:{server.server_port}"
    print(f"DeSynth Studio: {url}\nStop: Ctrl+C", flush=True)
    if not args.no_browser:
        threading.Timer(.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDeSynth Studio stopped.")
    finally:
        server.server_close()
        studio.close()


if __name__ == "__main__":
    main()
