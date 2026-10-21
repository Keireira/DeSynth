"""PyTorch/Diffusers inference only; the Rust application owns image processing."""
from __future__ import annotations

import gc
import logging
import math
import os
from pathlib import Path
import warnings

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
for message in (".*not writable.*", ".*pooled_projection_dim.*", ".*classifier-free guidance is not enabled.*"):
    warnings.filterwarnings("ignore", message=message)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)

import numpy as np
import torch
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    GGUFQuantizationConfig,
    QwenImageImg2ImgPipeline,
    QwenImageTransformer2DModel,
)
from PIL import Image

from diffusers_compat import apply_gguf_patches, load_cached

QWEN_IMAGE_REPO = "Qwen/Qwen-Image"
LIGHTNING_SCHEDULER_SHIFT = 3.0
EMBED_KEYS = ("prompt_embeds", "negative_prompt_embeds", "prompt_embeds_mask", "negative_prompt_embeds_mask")


def weight_identity(path: Path) -> tuple:
    stat = path.stat()
    if not path.is_file() or stat.st_size == 0:
        raise ValueError(f"Missing or empty model file: {path}")
    return (str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def effective_img2img_steps(steps: int, denoise: float) -> int:
    initial = min(steps * denoise, steps)
    return steps - int(max(steps - initial, 0))


def img2img_sigmas(steps: int, denoise: float):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if isinstance(denoise, bool) or not isinstance(denoise, (int, float)) or not math.isfinite(denoise) or not 0 < denoise <= 1:
        raise ValueError("denoise must be a finite number greater than zero and at most one")
    actual = effective_img2img_steps(steps, denoise)
    if actual < 1:
        raise ValueError("denoise and steps must produce at least one update")
    target = np.linspace(denoise, 0.0, actual + 1, dtype=np.float64)[:-1]
    return target / (LIGHTNING_SCHEDULER_SHIFT - (LIGHTNING_SCHEDULER_SHIFT - 1.0) * target)


def resolved_offload(device: str, offload: str) -> str:
    target = torch.device(device)
    if target.type not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unsupported device: {device}")
    if offload == "auto":
        offload = "sequential" if target.type == "cuda" else "model" if target.type == "mps" else "none"
    if offload not in ("none", "sequential", "model"):
        raise ValueError(f"Unsupported offload mode: {offload}")
    if target.type == "cpu" and offload != "none":
        raise ValueError("CPU processing requires offload=none.")
    if target.type == "mps" and offload == "sequential":
        raise ValueError("Sequential offload is not supported on MPS.")
    return offload


def place_pipeline(pipe, device: str, offload: str) -> None:
    target = torch.device(device)
    offload = resolved_offload(device, offload)
    if offload == "none":
        pipe.to(str(target))
    elif target.type == "cuda":
        method = pipe.enable_sequential_cpu_offload if offload == "sequential" else pipe.enable_model_cpu_offload
        method(gpu_id=target.index if target.index is not None else 0, device="cuda")
    else:
        pipe.enable_model_cpu_offload(device=target.type)


def build_pipeline(transformer: Path, lightning: Path, device: str, offload: str, precision: str, scale: float):
    target = torch.device(device)
    if precision not in ("float32", "bfloat16"):
        raise ValueError("Qwen-Image requires float32 or bfloat16 precision.")
    if target.type == "cpu" and precision != "float32":
        raise ValueError("CPU processing requires float32 precision.")
    offload = resolved_offload(device, offload)
    dtype = getattr(torch, precision)
    apply_gguf_patches()
    model = load_cached(
        QwenImageTransformer2DModel.from_single_file,
        pretrained_model_link_or_path_or_dict=str(transformer),
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        dtype=dtype,
        config=QWEN_IMAGE_REPO,
        subfolder="transformer",
    )
    vae = load_cached(
        AutoencoderKLQwenImage.from_pretrained,
        pretrained_model_name_or_path=QWEN_IMAGE_REPO,
        subfolder="vae",
        dtype=dtype,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_config({
        "base_image_seq_len": 256,
        "base_shift": math.log(LIGHTNING_SCHEDULER_SHIFT),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(LIGHTNING_SCHEDULER_SHIFT),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    })
    pipe = load_cached(
        QwenImageImg2ImgPipeline.from_pretrained,
        pretrained_model_name_or_path=QWEN_IMAGE_REPO,
        transformer=model,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        scheduler=scheduler,
        dtype=dtype,
    )
    from safetensors.torch import load_file

    pipe.load_lora_weights(load_file(str(lightning), device="cpu"), adapter_name="lightning")
    pipe.set_adapters(["lightning"], adapter_weights=[scale])
    place_pipeline(pipe, device, offload)
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def sample(pipe, image, embeds, *, denoise: float, steps: int, seed: int, callback_on_step_end):
    sigmas = img2img_sigmas(steps, denoise)
    return pipe(
        image=image,
        prompt_embeds=embeds["prompt_embeds"].unsqueeze(0),
        prompt_embeds_mask=embeds["prompt_embeds_mask"].unsqueeze(0),
        negative_prompt_embeds=embeds["negative_prompt_embeds"].unsqueeze(0),
        negative_prompt_embeds_mask=embeds["negative_prompt_embeds_mask"].unsqueeze(0),
        num_inference_steps=len(sigmas),
        sigmas=sigmas,
        true_cfg_scale=1.0,
        strength=1.0,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        callback_on_step_end=callback_on_step_end,
        width=image.width,
        height=image.height,
    ).images[0]


class Runtime:
    def __init__(self):
        self.pipe = None
        self.pipe_key = None
        self.embeds = None
        self.embeds_key = None

    def release(self):
        self.pipe = None
        self.pipe_key = None
        self.embeds = None
        self.embeds_key = None
        gc.collect()
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                with torch.cuda.device(index):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def run(self, request: dict, progress):
        transformer = Path(request["transformer"])
        lightning = Path(request["lightning"])
        embeds_path = Path(request["embeds"])
        base_key = (weight_identity(transformer), weight_identity(lightning), request["device"],
                    request["offload"], request["precision"])
        scale = request["lightning_scale"]
        if isinstance(scale, bool) or not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Lightning scale must be finite and greater than zero.")
        embeds_key = weight_identity(embeds_path)
        cached_embeds = self.embeds
        if cached_embeds is None or self.embeds_key != embeds_key:
            progress(3, "Loading embeddings")
            loaded = torch.load(embeds_path, map_location="cpu", weights_only=True)
            if not isinstance(loaded, dict) or any(not isinstance(loaded.get(key), torch.Tensor) for key in EMBED_KEYS):
                raise ValueError("Embeddings cache must contain all four prompt tensors.")
            cached_embeds = {key: loaded[key] for key in EMBED_KEYS}
        with Image.open(request["source"]) as source:
            image = source.convert("RGB")
        if image.width % 16 or image.height % 16:
            raise ValueError("Inference input dimensions must be multiples of 16.")
        key = (*base_key, scale)
        if self.pipe is None or self.pipe_key[:-1] != base_key:
            self.release()
            progress(7, f"Loading {transformer.name}")
            self.pipe = build_pipeline(transformer, lightning, request["device"], request["offload"],
                                       request["precision"], scale)
            self.pipe_key = key
        elif self.pipe_key != key:
            self.pipe.set_adapters(["lightning"], adapter_weights=[scale])
            self.pipe_key = key
        self.embeds = cached_embeds
        self.embeds_key = embeds_key
        dtype = getattr(torch, request["precision"])
        embeds = {key: value.to(device=request["device"], dtype=dtype if value.is_floating_point() else value.dtype)
                  for key, value in self.embeds.items()}
        current = image
        passes = request["passes"]
        actual_steps = effective_img2img_steps(request["steps"], request["denoise"])
        for pass_index in range(passes):
            def step_end(pipe, step, timestep, callback_kwargs):
                count = max(1, getattr(pipe, "num_timesteps", actual_steps))
                progress(min(88, 12 + 76 * (pass_index + (step + 1) / count) / passes),
                         f"Pass {pass_index + 1}/{passes} · step {step + 1}/{count}")
                return callback_kwargs

            progress(12 + 76 * pass_index / passes, f"Pass {pass_index + 1}/{passes} · encoding image")
            with torch.inference_mode():
                current = sample(self.pipe, current, embeds, denoise=request["denoise"], steps=request["steps"],
                                 seed=request["seed"] + pass_index, callback_on_step_end=step_end)
        output = Path(request["output"])
        temporary = output.with_name(f".{output.name}.tmp")
        try:
            current.save(temporary, format="PNG")
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        return {"output": str(output), "seed": request["seed"], "device": request["device"],
                "offload": request["offload"], "precision": request["precision"], "actual_steps": actual_steps}
