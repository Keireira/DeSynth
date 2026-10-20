"""Image preparation and output assembly shared by CLI and GUI."""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageOps


def load_rgba(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGBA")


def prepare_image(original: Image.Image, preview: bool = False) -> Image.Image:
    """Preserve aspect ratio and give the pipeline explicit multiples of 16."""
    image = original.convert("RGB")
    ratio = min(1.0, 768 / max(image.size)) if preview else 1.0
    # The epsilon prevents 768 / width * width rounding to 767.999999999.
    size = tuple(max(16, math.floor(value * ratio / 16 + 1e-9) * 16) for value in image.size)
    return image if image.size == size else image.resize(size, Image.Resampling.LANCZOS)


def validate_restoration(sigma: float, mode: str, unsharp: float) -> None:
    for value, low, high, name in ((sigma, 0.1, 10, "restore_sigma"), (unsharp, 0, 3, "unsharp")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be a finite number between {low} and {high}.")
    if mode not in ("gaussian", "edge"):
        raise ValueError("restore_mode must be gaussian or edge.")


def finish_image(clean: Image.Image, original: Image.Image, params: dict, restore_fn) -> Image.Image:
    """Restore detail when requested, retaining source dimensions and alpha."""
    validate_restoration(params["restore_sigma"], params["restore_mode"], params["unsharp"])
    if params["restore"]:
        final = restore_fn(clean, original, sigma=params["restore_sigma"], mode=params["restore_mode"],
                           unsharp_strength=params["unsharp"])
    else:
        final = clean.convert("RGB")
    if final.size != original.size:
        final = final.resize(original.size, Image.Resampling.LANCZOS)
    final = final.convert("RGBA")
    final.putalpha(original.convert("RGBA").getchannel("A"))
    return final
