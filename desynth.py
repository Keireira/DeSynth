# Qwen-Image GGUF img2img + frequency-domain detail restore. Removes SynthID
# from images while keeping them visually close to the source.

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Silence known-harmless noise from torch/diffusers/HF.
warnings.filterwarnings("ignore", message=".*not writable.*")
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
warnings.filterwarnings("ignore", message=".*pooled_projection_dim.*")
warnings.filterwarnings("ignore", message=".*classifier-free guidance is not enabled.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
try:
    from diffusers.utils import logging as _df_logging
    _df_logging.set_verbosity_error()
except Exception:
    pass

import torch

from diffusers_compat import apply_gguf_patches, load_cached
from processing import finish_image, load_rgba, prepare_image, validate_restoration
import model_catalog

from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    GGUFQuantizationConfig,
    QwenImageImg2ImgPipeline,
    QwenImageTransformer2DModel,
)
ROOT = Path(__file__).resolve().parent
AI_DIR = ROOT / "ai"
DEFAULT_INPUT = ROOT / "original.png"
OUT_DIR = ROOT / "out"

def _nonempty_file(path: Path) -> bool:
    return model_catalog.nonempty_file(path)


def resolve_model_path(path: str | Path) -> Path:
    """Resolve bare model filenames in ai/ first, retaining legacy root files."""
    return model_catalog.resolve_model_path(path, ROOT)


def discover_gguf_models() -> list[Path]:
    """Find all local GGUFs, preferring ai/ when a filename exists twice."""
    return model_catalog.discover_models(ROOT)


def discover_lightning_loras() -> list[Path]:
    """List local Qwen-Image-2512 Lightning adapters, newest step profile first."""
    return model_catalog.discover_adapters(ROOT)


def lightning_steps(path: str | Path) -> int | None:
    """Return the advertised training profile, not the img2img step count."""
    return model_catalog.lightning_steps(path)


def default_steps_for_lightning(path: str | Path) -> int:
    """Total schedule at default strength: 8 actual steps for the 8-step LoRA."""
    return model_catalog.default_steps_for_lightning(path)


def effective_img2img_steps(steps: int, denoise: float) -> int:
    """Match QwenImageImg2ImgPipeline.get_timesteps, including its rounding."""
    return model_catalog.effective_img2img_steps(steps, denoise)

# Default transformer = best available quant right now (used by --transformer default).
GGUF_TRANSFORMER = next(iter(discover_gguf_models()), AI_DIR / "Qwen-Image-2512-Q4_K_M.gguf")
LIGHTNING_LORA = next(iter(discover_lightning_loras()),
                      AI_DIR / "Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors")
DEFAULT_LIGHTNING_SCALE = 0.8
EMBEDS_CACHE = model_catalog.embeddings_path(ROOT)


def _is_oom(exc: BaseException) -> bool:
    """Return True for any out-of-memory error across CUDA, MPS, and CPU."""
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return any(k in msg for k in ("out of memory", "oom", "memory allocation"))
    # torch.cuda.OutOfMemoryError is a subclass of RuntimeError, already covered.
    return False


def _free_pipeline(pipe=None) -> None:
    """Collect after the caller has dropped its pipeline reference."""
    import gc
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        # GUI jobs can switch CUDA devices; the previous pipeline's allocator
        # may belong to a device other than the current default.
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# HF repo used only for tiny configs and the ~250MB VAE.
QWEN_IMAGE_REPO = "Qwen/Qwen-Image"

# Img2img slices the total schedule according to denoise (see the effective
# step helper above for the pipeline's rounding). The 8-step adapter
# therefore defaults to 32 total steps at strength 0.25. Keep the previous
# 8-total-step profile when only the legacy 4-step adapter is available.
# Explicit --steps values are always used as supplied.
DEFAULT_STEPS = default_steps_for_lightning(LIGHTNING_LORA)
CFG = 1.0
DEFAULT_DENOISE = 0.25
# SynthID's spatial cutoff is at ~2px. 1.95 is the safe ceiling that takes
# back maximum detail from the original without dragging the watermark along.
DEFAULT_RESTORE_SIGMA = 1.95

# ---------------------------------------------------------------------------
# Device / dtype selection — CUDA > MPS (Apple Silicon) > CPU
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
    COMPUTE_DTYPE = torch.bfloat16   # native on Ampere+
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    # PyTorch 2.14 has solid bfloat16 MPS coverage. More importantly, the
    # diffusers pipeline's image preprocessor naturally produces bfloat16
    # tensors on MPS — using float16 here causes a dtype mismatch in the
    # VAE encoder (Input bfloat16 vs bias float16). Keep bfloat16 throughout.
    COMPUTE_DTYPE = torch.bfloat16
    # Disable PyTorch's conservative MPS OOM guard (~70% threshold) so macOS's
    # own memory compressor and SSD swap can manage pressure instead.
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
else:
    DEVICE = "cpu"
    COMPUTE_DTYPE = torch.float32    # bf16/fp16 are slow on CPU without HW support

print(f"[device] {DEVICE}  dtype={COMPUTE_DTYPE}")


def _place_pipeline(pipe, device: str, offload: str = "auto") -> None:
    """Place a new pipeline without changing the CLI's process-wide defaults."""
    target = torch.device(device)
    if offload == "auto":
        offload = "sequential" if target.type == "cuda" else "model" if target.type == "mps" else "none"
    if offload not in ("none", "sequential", "model"):
        raise ValueError(f"Unsupported offload mode: {offload}")
    if target.type == "cpu" and offload != "none":
        raise ValueError("CPU processing requires offload=none.")
    if target.type == "mps" and offload == "sequential":
        raise ValueError("Sequential offload is not supported on MPS. Choose model or none.")
    if offload == "none":
        pipe.to(str(target))
    elif target.type == "cuda":
        method = pipe.enable_sequential_cpu_offload if offload == "sequential" else pipe.enable_model_cpu_offload
        method(gpu_id=target.index if target.index is not None else 0, device="cuda")
    else:
        pipe.enable_model_cpu_offload(device=target.type)


def build_pipeline(
    transformer_path: Path = GGUF_TRANSFORMER, *, device: str | None = None,
    offload: str = "auto", compute_dtype: torch.dtype | None = None,
    lightning_lora: Path | None = None, lightning_scale: float = DEFAULT_LIGHTNING_SCALE,
) -> QwenImageImg2ImgPipeline:
    transformer_path = resolve_model_path(transformer_path)
    lightning_lora = resolve_model_path(lightning_lora or LIGHTNING_LORA)
    if not math.isfinite(lightning_scale) or lightning_scale <= 0:
        raise ValueError("Lightning scale must be a finite number greater than zero.")
    target = torch.device(device or DEVICE)
    dtype = compute_dtype or (torch.float32 if target.type == "cpu" else COMPUTE_DTYPE)
    if target.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU processing requires float32 precision.")
    for p in (transformer_path, lightning_lora, EMBEDS_CACHE):
        if not _nonempty_file(p):
            raise FileNotFoundError(
                f"{p}\n(Run `python precompute_embeds.py` first if embeds_cache.pt is missing.)"
            )

    apply_gguf_patches()
    transformer = load_cached(
        QwenImageTransformer2DModel.from_single_file,
        pretrained_model_link_or_path_or_dict=str(transformer_path),
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

    # Both Lightning profiles are distilled with shift=3. Match their official
    # FlowMatch schedule, including disabling the base model's terminal shift.
    # https://github.com/ModelTC/LightX2V-Qwen-Image-Lightning/blob/main/generate_with_diffusers.py
    scheduler = FlowMatchEulerDiscreteScheduler.from_config({
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
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

    # No text encoder loaded — embeds are precomputed.
    pipe = load_cached(
        QwenImageImg2ImgPipeline.from_pretrained,
        pretrained_model_name_or_path=QWEN_IMAGE_REPO,
        transformer=transformer,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        scheduler=scheduler,
        dtype=dtype,
    )

    # Load LoRA via safetensors mmap so its 1.6 GB stays SSD-backed (same
    # strategy as the GGUF weights). safetensors.torch.load_file() memory-maps
    # the file by default; tensors are only paged into RAM when accessed.
    from safetensors.torch import load_file as _st_load
    _lora_sd = _st_load(str(lightning_lora), device="cpu")
    pipe.load_lora_weights(_lora_sd, adapter_name="lightning")
    pipe.set_adapters(["lightning"], adapter_weights=[lightning_scale])
    print(f"[lightning] {lightning_lora.name}  scale={lightning_scale:g}")

    _place_pipeline(pipe, str(target), offload)
    pipe.vae.enable_tiling()

    return pipe


def _split(arr, sigma):
    import cv2
    f = arr.astype("float32")
    low = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return low, f - low


# Gaussian: clean low band + original high band at one fixed sigma.
def _restore_gaussian(clean, original, sigma):
    low, _ = _split(clean, sigma)
    _, high = _split(original, sigma)
    return (low + high).clip(0, 255)


# Edge-aware: two recipes blended by edge mask. At edges, sigma=1.0 pulls
# more original detail back; in flats, sigma=1.95 keeps the watermark dead.
# Watermark survives only where edge mask -> 0, so flat skin/sky still clean.
def _restore_edge(clean, original, sigma_safe, sigma_edge=1.0):
    import cv2
    import numpy as np
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype("float32")
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), sigmaX=2.0, sigmaY=2.0)
    m = mag / (np.percentile(mag, 99) + 1e-6)
    m = np.clip(m, 0, 1)[..., None]
    safe = _restore_gaussian(clean, original, sigma_safe)
    sharp = _restore_gaussian(clean, original, sigma_edge)
    return (m * sharp + (1 - m) * safe).clip(0, 255)


def _restore(clean_pil, original_pil, *, sigma: float, mode: str = "gaussian", unsharp_strength: float = 0.0):
    import cv2
    import numpy as np
    from PIL import Image

    validate_restoration(sigma, mode, unsharp_strength)
    clean = np.asarray(clean_pil.convert("RGB"))
    original = np.asarray(original_pil.convert("RGB"))
    if clean.shape != original.shape:
        clean = cv2.resize(clean, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    if mode == "gaussian":
        combined = _restore_gaussian(clean, original, sigma)
    elif mode == "edge":
        combined = _restore_edge(clean, original, sigma)
    else:
        raise ValueError(f"unknown restore mode: {mode}")
    combined = combined.astype("float32")
    if unsharp_strength > 0:
        blurred = cv2.GaussianBlur(combined, (0, 0), sigmaX=1.0, sigmaY=1.0)
        combined = (combined + unsharp_strength * (combined - blurred)).clip(0, 255)
    return Image.fromarray(combined.astype("uint8"))


def _sample(pipe, image, embeds, *, denoise: float, steps: int, seed: int, callback_on_step_end=None, width=None, height=None):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    callbacks = {"callback_on_step_end": callback_on_step_end} if callback_on_step_end is not None else {}
    if width is not None and height is not None:
        callbacks.update(width=width, height=height)
    return pipe(
        image=image,
        prompt_embeds=embeds["prompt_embeds"].unsqueeze(0),
        prompt_embeds_mask=embeds["prompt_embeds_mask"].unsqueeze(0),
        negative_prompt_embeds=embeds["negative_prompt_embeds"].unsqueeze(0),
        negative_prompt_embeds_mask=embeds["negative_prompt_embeds_mask"].unsqueeze(0),
        num_inference_steps=steps,
        true_cfg_scale=CFG,
        strength=denoise,
        generator=generator,
        **callbacks,
    ).images[0]


def run_once(pipe, image, embeds, *, denoise: float, steps: int, seed: int, passes: int):
    current = image
    for i in range(passes):
        # Reseed per pass so iterations aren't identical perturbations.
        with torch.inference_mode():
            current = _sample(pipe, current, embeds, denoise=denoise, steps=steps, seed=seed + i,
                              width=image.width, height=image.height)
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT), help="input image path")
    parser.add_argument("--denoise", type=float, nargs="+", default=[DEFAULT_DENOISE], help="denoise strength(s)")
    parser.add_argument("--steps", type=int, default=None,
                        help="total img2img schedule steps; default 32 for 8-step Lightning, 8 for 4-step")
    parser.add_argument("--passes", type=int, default=1, help="sequential img2img passes")
    parser.add_argument("--seed", type=int, default=None, help="random per run if omitted")
    parser.add_argument("--restore-sigma", type=float, default=DEFAULT_RESTORE_SIGMA, help="freq-restore sigma; cliff at 2.0")
    parser.add_argument("--restore-mode", choices=["gaussian", "edge"], default="gaussian", help="restore recipe")
    parser.add_argument("--unsharp", type=float, default=0.0, help="post-restore sharpening; try 0.2 for perceptual lift")
    parser.add_argument("--no-restore", dest="restore", action="store_false", help="skip frequency restore")
    parser.add_argument("--keep-intermediate", action="store_true", help="also save the pre-restore output")
    parser.add_argument("--transformer", type=Path, default=None, help="GGUF path; bare filenames resolve in ai/ first")
    parser.add_argument("--lightning", type=Path, default=LIGHTNING_LORA,
                        help="Lightning LoRA path; defaults to local 8-step adapter, then legacy 4-step")
    parser.add_argument("--lightning-scale", type=float, default=DEFAULT_LIGHTNING_SCALE, help="Lightning LoRA strength (default 0.8)")
    parser.set_defaults(restore=True)
    args = parser.parse_args()

    args.lightning = resolve_model_path(args.lightning)
    if args.steps is None:
        args.steps = default_steps_for_lightning(args.lightning)
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.passes < 1:
        parser.error("--passes must be at least 1")
    if args.seed is not None and not 0 <= args.seed <= 2**63 - args.passes:
        parser.error("--seed must be non-negative and leave room for each pass below 2**63")
    try:
        validate_restoration(args.restore_sigma, args.restore_mode, args.unsharp)
    except ValueError as error:
        parser.error(str(error))
    if not math.isfinite(args.lightning_scale) or args.lightning_scale <= 0:
        parser.error("--lightning-scale must be a finite number greater than zero")
    if not _nonempty_file(args.lightning):
        parser.error(f"Lightning adapter not found: {args.lightning}")
    if any(not 0 < strength <= 1 for strength in args.denoise):
        parser.error("--denoise values must be greater than zero and at most 1")
    if any(effective_img2img_steps(args.steps, strength) < 1 for strength in args.denoise):
        parser.error("--steps * --denoise must execute at least one denoising step")
    print(f"[schedule] {args.steps} total steps; actual per pass: "
          + ", ".join(f"{effective_img2img_steps(args.steps, strength)} at denoise={strength:g}" for strength in args.denoise))

    if args.seed is None:
        import secrets
        args.seed = secrets.randbelow(2**63 - args.passes + 1)
        print(f"seed: {args.seed} (random)")
    else:
        print(f"seed: {args.seed} (fixed)")

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"missing {input_path}")

    original = load_rgba(input_path)
    image = prepare_image(original)
    print(f"input: {input_path.name}  size={original.size}  processing={image.size}")

    cached_embeds = torch.load(EMBEDS_CACHE, map_location="cpu", weights_only=False)
    embeds = {}
    for key in ("prompt_embeds", "negative_prompt_embeds", "prompt_embeds_mask", "negative_prompt_embeds_mask"):
        value = cached_embeds[key]
        embeds[key] = value.to(device=DEVICE, dtype=COMPUTE_DTYPE if value.is_floating_point() else value.dtype)
    del cached_embeds
    print(f"embeds: pos={tuple(embeds['prompt_embeds'].shape)} neg={tuple(embeds['negative_prompt_embeds'].shape)}")

    # Build the fallback sequence to attempt.
    # If --transformer was given explicitly, respect it and skip auto-fallback.
    explicit_transformer = args.transformer is not None
    if explicit_transformer:
        args.transformer = resolve_model_path(args.transformer)
        attempt_chain = [(args.transformer.stem, args.transformer)]
    else:
        models = [{"path": str(path.relative_to(ROOT))} for path in discover_gguf_models()]
        attempt_chain = [(Path(path).stem, ROOT / path)
                         for path in model_catalog.automatic_model_paths(models, ROOT)]
        if not attempt_chain:
            raise SystemExit(
                "No GGUF transformer found in ai/ or the project root. Download at least one quant:\n"
                "  Qwen-Image-2512-Q4_K_M.gguf  (~13 GB, best quality)\n"
                "  Qwen-Image-2512-Q2_K_M.gguf  (~7 GB,  fallback)\n"
                "from https://huggingface.co/Frederic75/Qwen-Image-2512-GGUF"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex[:12]
    stem = input_path.stem[:60]
    params = {key: getattr(args, key) for key in (
        "steps", "passes", "seed", "restore", "restore_sigma", "restore_mode", "unsharp", "keep_intermediate",
    )}
    manifest = {
        "schema_version": 1, "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": dict(path=str(input_path), name=input_path.name, width=original.width, height=original.height),
        "working_size": list(image.size), "params": params,
        "lightning": dict(path=str(args.lightning.resolve()), name=args.lightning.name, scale=args.lightning_scale),
        "runtime": dict(device=DEVICE, precision=str(COMPUTE_DTYPE), offload="auto"),
        "outputs": [],
    }
    manifest_path = OUT_DIR / f"{stem}_desynth_{run_id}.json"

    def save_result(clean_pil, denoise, index, model_path):
        # Readable parameter tags plus a run ID and sweep index prevent both
        # cross-run collisions and repeated strengths within a single sweep.
        quant = re.search(r"(?:^|-)([qi]\d[\w]*)$", model_path.stem, re.IGNORECASE)
        model_tag = (quant.group(1).lower() if quant else re.sub(r"[^\w-]+", "_", model_path.stem))[:32]
        precision = re.search(r"-(fp32|bf16)(?:\.|$)", args.lightning.name, re.IGNORECASE)
        adapter_tag = f"l{lightning_steps(args.lightning) or 'custom'}" + (precision.group(1).lower() if precision else "")
        tag = (f"{model_tag}_{adapter_tag}_s{args.steps}_d{denoise:g}_p{args.passes}_seed{args.seed}"
               f"_r{args.restore_sigma:g}{args.restore_mode}_u{args.unsharp:g}")
        if not args.restore:
            tag += "_norestore"
        prefix = f"{stem}_desynth_{tag}_{run_id}_{index:02d}"
        final_path = OUT_DIR / f"{prefix}.png"
        finish_image(clean_pil, original, params, _restore).save(final_path)
        print(f"  -> {final_path}")
        mid_path = None
        if args.restore and args.keep_intermediate:
            mid_path = OUT_DIR / f"{prefix}_clean.png"
            finish_image(clean_pil, original, {**params, "restore": False}, _restore).save(mid_path)
            print(f"  -> {mid_path}  (intermediate)")
        manifest["outputs"].append(dict(
            kind="result", path=str(final_path.resolve()), name=final_path.name, denoise=denoise,
            effective_steps=effective_img2img_steps(args.steps, denoise),
            clean_path=str(mid_path.resolve()) if mid_path else None,
        ))
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)

    pipe = None
    try:
        first_clean = None
        used_path = None
        for attempt, (label, transformer_path) in enumerate(attempt_chain):
            print(f"[quant] trying {label}  ({transformer_path.name})")
            try:
                pipe = build_pipeline(transformer_path=transformer_path,
                                      lightning_lora=args.lightning, lightning_scale=args.lightning_scale)
                first_clean = run_once(pipe, image, embeds, denoise=args.denoise[0],
                                       steps=args.steps, seed=args.seed, passes=args.passes)
                used_path = transformer_path
                break
            except Exception as exc:
                if not (_is_oom(exc) and not explicit_transformer and attempt + 1 < len(attempt_chain)):
                    raise
                print(f"[OOM] {label} ran out of memory — retrying with {attempt_chain[attempt + 1][0]}")
            # Leaving the exception scope also releases the failed call's
            # traceback (which may hold transformer tensors). Drop the last
            # pipeline reference before collecting and loading the next model.
            pipe = None
            _free_pipeline()

        if first_clean is None or used_path is None:
            raise RuntimeError("All quants exhausted without success.")
        manifest["transformer"] = dict(path=str(used_path.resolve()), name=used_path.name)
        save_result(first_clean, args.denoise[0], 1, used_path)
        del first_clean
        for index, denoise in enumerate(args.denoise[1:], start=2):
            clean = run_once(pipe, image, embeds, denoise=denoise, steps=args.steps,
                             seed=args.seed, passes=args.passes)
            save_result(clean, denoise, index, used_path)
        print(f"  -> {manifest_path}  (run settings)")
    finally:
        pipe = None
        embeds.clear()
        _free_pipeline()


if __name__ == "__main__":
    main()
