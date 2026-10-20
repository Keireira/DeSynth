"""Image preparation and output assembly shared by CLI and GUI."""
from __future__ import annotations

import math
import io
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageOps


@lru_cache(maxsize=1)
def srgb_profile_bytes() -> bytes:
    """A stable sRGB tag lets saved files bypass repeated color transforms."""
    from PIL import ImageCms

    profile = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    # ICC creation date is irrelevant to the transform. Fix it so the profile
    # remains identical after restarting, and leave its optional profile ID unset.
    profile[24:36] = bytes.fromhex("07d0 0001 0001 0000 0000 0000")
    profile[84:100] = bytes(16)
    return bytes(profile)


def oriented_size(image: Image.Image) -> tuple[int, int]:
    """Display dimensions without decoding an entire image just to register it."""
    return image.size[::-1] if image.getexif().get(274) in (5, 6, 7, 8) else image.size


def normalize_rgba(image: Image.Image) -> Image.Image:
    """Orient pixels and convert tagged colors to sRGB, retaining exact alpha."""
    oriented = ImageOps.exif_transpose(image)
    profile = oriented.info.get("icc_profile")
    if profile is None:
        result = oriented.convert("RGBA")
        result.info.pop("transparency", None)
        return result
    from PIL import ImageCms

    try:
        if not isinstance(profile, bytes) or not profile:
            raise ValueError("the embedded profile is empty or invalid")
        target = srgb_profile_bytes()
        if profile == target and oriented.mode in ("RGB", "RGBA"):
            result = oriented.convert("RGBA")
            result.info.pop("transparency", None)
            return result
        source_profile = ImageCms.ImageCmsProfile(io.BytesIO(profile))
        alpha = (oriented.getchannel("A") if oriented.mode in ("RGBA", "LA", "PA")
                 else oriented.convert("RGBA").getchannel("A") if oriented.mode == "P" or "transparency" in oriented.info
                 else Image.new("L", oriented.size, 255))
        # Palette transparency is retained above; the profile describes its RGB
        # colors. CMYK and Lab must reach LittleCMS before any naive RGB convert.
        color = (oriented if oriented.mode in ("RGB", "CMYK", "LAB", "L")
                 else oriented.convert("L" if oriented.mode == "LA" else "RGB"))
        result = ImageCms.profileToProfile(color, source_profile,
                                          ImageCms.ImageCmsProfile(io.BytesIO(target)), outputMode="RGB")
        result.putalpha(alpha)
        result.info.update(oriented.info)
        result.info["icc_profile"] = target
        result.info.pop("transparency", None)
        return result
    except (ImageCms.PyCMSError, OSError, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert embedded ICC color profile to sRGB: {exc}") from exc


def load_rgba(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return normalize_rgba(image)


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
    if original.info.get("icc_profile") is not None and original.info["icc_profile"] == srgb_profile_bytes():
        final.info["icc_profile"] = original.info["icc_profile"]
    return final
