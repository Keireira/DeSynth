"""Portable environment setup, model downloads, and readiness checks."""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "ai"
QWEN_REPO = "Qwen/Qwen-Image"
GGUF_MODELS = {
    "q4": ("Qwen-Image-2512-Q4_K_M.gguf", "qwen-image-2512-Q4_K_M.gguf"),
    "q8": ("Qwen-Image-2512-Q8_0.gguf", "qwen-image-2512-Q8_0.gguf"),
}
CACHE_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "transformer/config.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)


def run(*command: str) -> None:
    print(f"> {subprocess.list2cmdline(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def env_python(directory: Path) -> Path:
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def require_env(directory: Path) -> Path:
    python = env_python(directory)
    if not (directory / "pyvenv.cfg").is_file() or not python.is_file():
        raise RuntimeError(f"No virtual environment at {directory}. Run make install first.")
    return python


def create_env(directory: Path) -> None:
    if directory.exists():
        require_env(directory)
        print(f"Using environment: {directory}", flush=True)
    else:
        print(f"Creating environment: {directory}", flush=True)
        venv.EnvBuilder(with_pip=True).create(directory)


def install(directory: Path, backend: str) -> None:
    create_env(directory)
    python = str(require_env(directory))
    requirements = "requirements.txt" if backend == "cuda" else "requirements_mac.txt"
    run(python, "-m", "pip", "install", "--upgrade", "pip")
    run(python, "-m", "pip", "install", "-r", str(ROOT / requirements))
    run(python, "-m", "pip", "check")


def present(name: str) -> bool:
    return find_asset(name) is not None


def find_asset(name: str) -> Path | None:
    """Prefer ai/, retaining compatibility with existing root-level assets."""
    return next((path for path in (MODEL_DIR / name, ROOT / name)
                 if path.is_file() and path.stat().st_size > 0), None)


def installed_gguf() -> Path | None:
    for directory in (MODEL_DIR, ROOT):
        if directory.is_dir():
            found = next((path for path in sorted(directory.glob("*.gguf"))
                          if path.is_file() and path.stat().st_size > 0), None)
            if found is not None:
                return found
    return None


def selected_gguf(model: str) -> Path | None:
    if model == "auto":
        return installed_gguf()
    return next((find_asset(name) for name in GGUF_MODELS[model] if present(name)), None)


def choose_download_model(model: str) -> str:
    """Ask only for an unspecified, missing GGUF on an interactive terminal."""
    if model != "auto":
        return model
    if sys.stdin is None or not sys.stdin.isatty():
        return "q4"
    print("Choose a GGUF model to download:\n"
          "  1. Q4_K_M (~13 GB) - lower memory use [default]\n"
          "  2. Q8_0 (~21.8 GB) - higher memory use", flush=True)
    while True:
        try:
            choice = input("Choice [1]: ").strip().casefold()
        except (EOFError, OSError):
            print("Using Q4_K_M.", flush=True)
            return "q4"
        if choice in ("", "1", "q4", "q4_k_m"):
            return "q4"
        if choice in ("2", "q8", "q8_0"):
            return "q8"
        print("Enter 1 for Q4_K_M or 2 for Q8_0.", flush=True)


def lightning_candidates(choice: str) -> tuple[str, ...]:
    steps = (8, 4) if choice == "auto" else (int(choice),)
    return tuple(f"Qwen-Image-2512-Lightning-{count}steps-V1.0-{precision}.safetensors"
                 for count in steps for precision in ("fp32", "bf16"))


def require_embeds() -> None:
    if not present("embeds_cache.pt"):
        raise RuntimeError("Missing embeds_cache.pt in ai/ or the repository root. Restore the included file.")


def download_models(lightning: str = "auto", model: str = "auto") -> None:
    from huggingface_hub import HfApi, hf_hub_download

    require_embeds()
    gguf = selected_gguf(model)
    if gguf is None:
        model = choose_download_model(model)
    models = (
        ("Frederic75/Qwen-Image-2512-GGUF", GGUF_MODELS["q4" if model == "auto" else model], gguf),
        ("lightx2v/Qwen-Image-2512-Lightning", lightning_candidates(lightning),
         next((find_asset(name) for name in lightning_candidates(lightning) if present(name)), None)),
    )
    for repo, names, existing in models:
        if existing:
            print(f"Already present: {existing}", flush=True)
            continue
        # Resolve the upstream spelling rather than assuming filename case.
        available = HfApi().list_repo_files(repo_id=repo)
        by_name = {name.casefold(): name for name in available}
        filename = next((by_name[name.casefold()] for name in names if name.casefold() in by_name), None)
        if filename is None:
            raise RuntimeError(f"Cannot find {names[0]} in {repo}.")
        print(f"Downloading {repo}/{filename}", flush=True)
        MODEL_DIR.mkdir(exist_ok=True)
        hf_hub_download(repo_id=repo, filename=filename, local_dir=str(MODEL_DIR))
    print("Model files are ready.", flush=True)


def cached_file(filename: str) -> Path | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return Path(hf_hub_download(QWEN_REPO, filename, local_files_only=True))
    except LocalEntryNotFoundError:
        return None


def prepare() -> None:
    from huggingface_hub import hf_hub_download

    for filename in CACHE_FILES:
        if cached_file(filename) is not None:
            print(f"Cached: {filename}", flush=True)
        else:
            print(f"Caching {QWEN_REPO}/{filename}", flush=True)
            hf_hub_download(QWEN_REPO, filename)
    print("VAE and configs are cached; transformer/text encoder weights were not requested.", flush=True)


def doctor(backend: str, lightning: str = "auto", model: str = "auto") -> None:
    print(f"Python {sys.version.split()[0]}: {sys.executable}")
    print(f"Dependency backend: {backend}")
    failures = []
    for module in (
        "torch", "torchvision", "diffusers", "transformers", "accelerate", "peft",
        "gguf", "huggingface_hub", "sentencepiece", "google.protobuf", "PIL",
        "safetensors", "numpy", "cv2", "skimage.metrics",
    ):
        try:
            importlib.import_module(module)
        except Exception as exc:
            failures.append(f"Import {module}: {exc}")
    try:
        import diffusers

        for name in (
            "AutoencoderKLQwenImage", "GGUFQuantizationConfig",
            "QwenImageImg2ImgPipeline", "QwenImageTransformer2DModel",
        ):
            getattr(diffusers, name)
    except Exception as exc:
        failures.append(f"Qwen pipeline imports: {exc}")
    try:
        import torch

        print(f"PyTorch {torch.__version__}; CUDA runtime: {torch.version.cuda}")
        if backend == "cuda":
            if not torch.cuda.is_available():
                failures.append("CUDA is unavailable. Check the NVIDIA driver, or use BACKEND=cpu.")
            else:
                print(f"Device: {torch.cuda.get_device_name(0)}")
        else:
            if torch.version.cuda is not None:
                failures.append("CUDA wheel installed for CPU backend. Run make install BACKEND=cpu.")
            print(f"Device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    except Exception as exc:
        failures.append(f"PyTorch device check: {exc}")
    gguf = selected_gguf(model)
    if gguf is None:
        failures.append(f"Missing GGUF model ({model}). Run make models MODEL={model}.")
    else:
        print(f"GGUF: {gguf}")
    adapter = next((find_asset(name) for name in lightning_candidates(lightning) if present(name)), None)
    if adapter is None:
        failures.append(f"Missing Lightning LoRA ({lightning}). Run make models LIGHTNING={lightning}.")
    else:
        print(f"Lightning: {adapter}")
    try:
        require_embeds()
    except RuntimeError as exc:
        failures.append(str(exc))
    try:
        for filename in CACHE_FILES:
            if cached_file(filename) is None:
                failures.append(f"Missing cached {filename}. Run make prepare.")
    except Exception as exc:
        failures.append(f"Hugging Face cache: {exc}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", flush=True)
        raise RuntimeError("Readiness check failed. Run make setup to prepare this environment.")
    print("Ready: dependencies, device, model files, and VAE/config cache are available.", flush=True)


def worker(command: str, directory: Path, backend: str, lightning: str = "auto", model: str = "auto") -> None:
    # Compare prefixes, not executable paths: POSIX venv interpreters can be symlinks.
    if Path(sys.prefix).resolve() != directory:
        run(str(require_env(directory)), str(Path(__file__).resolve()), command,
            "--venv", str(directory), "--backend", backend, "--lightning", lightning, "--model", model)
    elif command == "models":
        download_models(lightning, model)
    elif command == "prepare":
        prepare()
    else:
        doctor(backend, lightning, model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("help", "venv", "install", "models", "prepare", "doctor", "setup"))
    parser.add_argument("--venv", type=Path, default=Path(".venv"))
    parser.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--model", choices=("auto", "q4", "q8"), default="auto",
                        help="GGUF to prepare/check; auto reuses installed weights, otherwise asks in a terminal (Q4 default)")
    parser.add_argument("--lightning", choices=("auto", "4", "8"), default="auto",
                        help="adapter to prepare/check; auto reuses 8-step, then 4-step, or downloads 8-step FP32")
    args = parser.parse_args()
    if args.command == "help":
        parser.print_help()
        return
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        parser.exit(1, "Use Python 3.11 (supported: 3.10 through 3.13) for the pinned PyTorch release.\n")
    directory = (ROOT / args.venv).resolve()
    backend = args.backend
    if backend == "auto":
        backend = "cpu" if sys.platform == "darwin" else "cuda"
    if backend == "cuda" and sys.platform == "darwin":
        parser.exit(1, "CUDA is unavailable on macOS. Use --backend cpu for MPS/CPU.\n")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    try:
        if args.command == "venv":
            create_env(directory)
        elif args.command == "install":
            install(directory, backend)
        elif args.command == "setup":
            install(directory, backend)
            for command in ("models", "prepare", "doctor"):
                worker(command, directory, backend, args.lightning, args.model)
        else:
            worker(args.command, directory, backend, args.lightning, args.model)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    except Exception as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
