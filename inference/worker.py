"""Line-delimited JSON bridge. No HTTP, asset catalog, or image restoration."""
from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback

# Libraries can log to stdout; the executable reserves this stream for frames.
PROTOCOL = sys.stdout
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

MAX_COMMAND_BYTES = 65536
DEPENDENCIES = ("torch", "diffusers", "transformers", "accelerate", "peft", "gguf", "safetensors", "numpy", "PIL")
CACHE_FILES = ("model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
               "transformer/config.json", "vae/config.json", "vae/diffusion_pytorch_model.safetensors")


def emit(identifier, event, **fields):
    PROTOCOL.write(json.dumps({"id": identifier, "event": event, **fields}, allow_nan=False) + "\n")
    PROTOCOL.flush()


def probe():
    devices = [{"id": "cpu", "name": "CPU", "type": "cpu", "precisions": ["float32"], "offload_modes": ["none"]}]
    missing = [name for name in DEPENDENCIES if importlib.util.find_spec(name) is None]
    error = None
    torch_version = None
    cuda_runtime = None
    if "torch" not in missing:
        try:
            import torch

            torch_version = torch.__version__
            cuda_runtime = torch.version.cuda
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
                    tensor = torch.zeros(1, device="mps", dtype=torch.bfloat16)
                    del tensor
                    precisions.insert(0, "bfloat16")
                except (RuntimeError, TypeError):
                    pass
                devices.append({"id": "mps", "name": "Apple Silicon", "type": "mps", "precisions": precisions,
                                "offload_modes": ["model", "none"]})
        except Exception as exc:
            error = str(exc)
            missing.append("working PyTorch runtime")
    default = next((item for item in devices if item["type"] == "cuda"),
                   next((item for item in devices if item["type"] == "mps"), devices[0]))
    runtime = {"ready": True, "refreshing": False, "devices": devices, "default_device": default["id"]}
    if error is not None:
        runtime["error"] = error
    device = "CPU" if default["type"] == "cpu" else f"{default['type'].upper()} · {default['name']}"
    return {"ready": not missing, "missing": missing, "runtime": runtime, "device": device,
            "torch_version": torch_version, "cuda_runtime": cuda_runtime}


def doctor():
    result = probe()
    errors = []
    for module in (*DEPENDENCIES, "torchvision", "huggingface_hub", "sentencepiece", "google.protobuf"):
        try:
            importlib.import_module(module)
        except Exception as error:
            errors.append(f"Import {module}: {error}")
    try:
        import diffusers

        for name in ("AutoencoderKLQwenImage", "GGUFQuantizationConfig", "QwenImageImg2ImgPipeline",
                     "QwenImageTransformer2DModel"):
            getattr(diffusers, name)
        from diffusers_compat import apply_gguf_patches

        apply_gguf_patches()
    except Exception as error:
        errors.append(f"Qwen pipeline: {error}")
    result["errors"] = errors
    result["ready"] = result["ready"] and not errors
    return result


def validate_sample(request):
    for field in ("transformer", "lightning", "embeds", "source", "output"):
        value = request.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"{field} must be an absolute path")
        if field != "output" and (not Path(value).is_file() or Path(value).stat().st_size == 0):
            raise ValueError(f"Missing or empty {field} file: {value}")
    if not Path(request["output"]).parent.is_dir():
        raise ValueError("Output directory does not exist")
    for field, low, high in (("steps", 1, 1000), ("passes", 1, 20)):
        value = request.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{field} must be an integer between {low} and {high}")
    seed = request.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - request["passes"]:
        raise ValueError("seed must be nonnegative and leave room for every pass below 2**63")
    denoise = request.get("denoise")
    if isinstance(denoise, bool) or not isinstance(denoise, (int, float)) or not math.isfinite(denoise) or not 0 < denoise <= 1:
        raise ValueError("denoise must be finite, greater than zero, and at most one")
    if request["steps"] - int(max(request["steps"] - min(request["steps"] * denoise, request["steps"]), 0)) < 1:
        raise ValueError("denoise and steps must produce at least one update")
    scale = request.get("lightning_scale")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Lightning scale must be finite and greater than zero")
    if request.get("precision") not in ("float32", "bfloat16"):
        raise ValueError("precision must be float32 or bfloat16")
    if request.get("offload") not in ("none", "model", "sequential"):
        raise ValueError("offload must be none, model, or sequential")
    device = request.get("device")
    if not isinstance(device, str) or not (device in ("cpu", "mps") or
                                           device.startswith("cuda:") and device[5:].isdigit()):
        raise ValueError("device must be cpu, mps, or cuda:N")
    if device == "cpu" and (request["offload"] != "none" or request["precision"] != "float32"):
        raise ValueError("CPU processing requires offload=none and float32 precision")
    if device == "mps" and request["offload"] == "sequential":
        raise ValueError("Sequential offload is not supported on MPS")


def is_oom(error):
    if isinstance(error, MemoryError):
        return True
    if not isinstance(error, RuntimeError):
        return False
    message = str(error).lower()
    return "out of memory" in message or "memory allocation" in message or re.search(r"\boom\b", message) is not None


def main():
    runtime = None
    while True:
        line = sys.stdin.buffer.readline(MAX_COMMAND_BYTES + 1)
        if not line:
            break
        identifier = None
        oom = False
        error_fields = None
        try:
            if len(line) > MAX_COMMAND_BYTES:
                raise ValueError("Command exceeds 64 KiB")
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Command must be an object")
            identifier = request.get("id")
            if isinstance(identifier, bool) or not isinstance(identifier, int) or identifier < 0:
                raise ValueError("Command id must be a nonnegative integer")
            operation = request.get("op")
            if operation == "probe":
                result = probe()
            elif operation == "doctor":
                result = doctor()
            elif operation == "release":
                if runtime is not None:
                    runtime.release()
                result = {"released": True}
            elif operation == "prepare":
                from huggingface_hub import hf_hub_download
                from huggingface_hub.errors import LocalEntryNotFoundError

                for index, filename in enumerate(CACHE_FILES):
                    emit(identifier, "progress", progress=100 * index / len(CACHE_FILES), stage=f"Caching {filename}")
                    try:
                        hf_hub_download("Qwen/Qwen-Image", filename, local_files_only=True)
                    except LocalEntryNotFoundError:
                        hf_hub_download("Qwen/Qwen-Image", filename)
                result = {"cached": list(CACHE_FILES)}
            elif operation == "sample":
                validate_sample(request)
                emit(identifier, "progress", progress=2, stage="Loading inference libraries")
                if runtime is None:
                    from runtime import Runtime

                    runtime = Runtime()
                result = runtime.run(request, lambda progress, stage: emit(
                    identifier, "progress", progress=progress, stage=stage))
            else:
                raise ValueError(f"Unsupported inference operation: {operation}")
            emit(identifier, "result", result=result)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            oom = is_oom(error)
            error_fields = {"error": str(error), "oom": oom}
        # Leave the exception scope before freeing a failed pipeline: tracebacks
        # can retain multi-gigabyte tensors until this point.
        if error_fields is not None:
            if oom and runtime is not None:
                runtime.release()
            emit(identifier, "error", **error_fields)
    if runtime is not None:
        runtime.release()


if __name__ == "__main__":
    sys.stdout = sys.stderr
    main()
