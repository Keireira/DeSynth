"""Local model discovery shared by setup, the CLI, and the browser server.

This module uses only the standard library so setup can import it before the
virtual environment or machine-learning dependencies exist.
"""
from __future__ import annotations

import re
from pathlib import Path


def nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except FileNotFoundError:
        return False


def find_asset(name: str, root: Path) -> Path | None:
    for directory in (root / "ai", root):
        exact = directory / name
        if nonempty_file(exact):
            return exact
        if directory.is_dir():
            for path in sorted(directory.iterdir()):
                if path.name.casefold() == name.casefold() and nonempty_file(path):
                    return path
    return None


def _files(root: Path, suffix: str) -> list[Path]:
    found: dict[str, Path] = {}
    for directory in (root / "ai", root):
        if directory.is_dir():
            for path in sorted(directory.iterdir()):
                if path.suffix.casefold() == suffix and nonempty_file(path):
                    found.setdefault(path.name.casefold(), path)
    return list(found.values())


def discover_models(root: Path) -> list[Path]:
    def priority(path):
        name = path.name.casefold()
        return (0 if "q4_k_m" in name else 1 if "q2_k_m" in name else 2,
                name)
    return sorted(_files(root, ".gguf"), key=priority)


def lightning_steps(path: str | Path) -> int | None:
    match = re.search(r"-(4|8)steps(?:-|\.|$)", Path(path).name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def discover_adapters(root: Path) -> list[Path]:
    paths = [path for path in _files(root, ".safetensors")
             if "qwen-image-2512-lightning-" in path.name.casefold()
             and lightning_steps(path) in (4, 8)]
    return sorted(paths, key=lambda path: (
        lightning_steps(path) != 8, "fp32" not in path.name.casefold(),
        path.name.casefold(),
    ))


def resolve_model_path(path: str | Path, root: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    if len(path.parts) == 1:
        return find_asset(path.name, root) or root / "ai" / path
    return path.resolve() if nonempty_file(path) else root / path


def embeddings_path(root: Path) -> Path:
    return find_asset("embeds_cache.pt", root) or root / "embeds_cache.pt"


def automatic_model_paths(models: list[dict], root: Path) -> list[str]:
    """Use the preferred model, then only smaller files on out-of-memory."""
    available = []
    for model in models:
        path = root / model["path"]
        try:
            if path.is_file() and (size := path.stat().st_size) > 0:
                available.append((model["path"], size))
        except FileNotFoundError:
            continue
    if not available:
        return []
    preferred, preferred_size = available[0]
    return [preferred] + [path for path, size in sorted(
        available[1:], key=lambda item: item[1], reverse=True,
    ) if size < preferred_size]


def default_steps_for_lightning(path: str | Path) -> int:
    return 32 if lightning_steps(path) == 8 else 8


def effective_img2img_steps(steps: int, denoise: float) -> int:
    initial = min(steps * denoise, steps)
    return steps - int(max(steps - initial, 0))
