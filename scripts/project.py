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
QWEN_REPO = "Qwen/Qwen-Image"
GGUF_NAMES = (
    "Qwen-Image-2512-Q4_K_M.gguf",
    "qwen-image-2512-Q4_K_M.gguf",
    "Qwen-Image-2512-Q2_K_M.gguf",
    "qwen-image-2512-Q2_K_M.gguf",
)
LORA_NAME = "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors"
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
    path = ROOT / name
    return path.is_file() and path.stat().st_size > 0


def require_embeds() -> None:
    if not present("embeds_cache.pt"):
        raise RuntimeError("Missing embeds_cache.pt. Restore the included file from the repository.")


def download_models() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    require_embeds()
    models = (
        ("Frederic75/Qwen-Image-2512-GGUF", GGUF_NAMES[:2]),
        ("lightx2v/Qwen-Image-2512-Lightning", (LORA_NAME,)),
    )
    for repo, names in models:
        existing = next((name for name in names if present(name)), None)
        if existing:
            print(f"Already present: {existing}", flush=True)
            continue
        # Resolve the upstream spelling rather than assuming filename case.
        available = HfApi().list_repo_files(repo_id=repo)
        expected = {name.casefold() for name in names}
        filename = next((name for name in available if name.casefold() in expected), None)
        if filename is None:
            raise RuntimeError(f"Cannot find {names[0]} in {repo}.")
        print(f"Downloading {repo}/{filename}", flush=True)
        hf_hub_download(repo_id=repo, filename=filename, local_dir=str(ROOT))
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


def doctor(backend: str, lightning: str = "auto") -> None:
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
    gguf = installed_gguf()
    if gguf is None:
        failures.append("Missing GGUF model. Run make models.")
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


def worker(command: str, directory: Path, backend: str, lightning: str = "auto") -> None:
    # Compare prefixes, not executable paths: POSIX venv interpreters can be symlinks.
    if Path(sys.prefix).resolve() != directory:
        run(str(require_env(directory)), str(Path(__file__).resolve()), command,
            "--venv", str(directory), "--backend", backend, "--lightning", lightning)
    elif command == "models":
        download_models(lightning)
    elif command == "prepare":
        prepare()
    else:
        doctor(backend, lightning)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("help", "venv", "install", "models", "prepare", "doctor", "setup"))
    parser.add_argument("--venv", type=Path, default=Path(".venv"))
    parser.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="auto")
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
                worker(command, directory, backend, args.lightning)
        else:
            worker(args.command, directory, backend, args.lightning)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    except Exception as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
