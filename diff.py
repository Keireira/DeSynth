"""Visualize absolute differences across all four RGBA channels."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


def positive_gain(value: str) -> float:
    try:
        gain = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("gain must be a finite positive number") from exc
    if not math.isfinite(gain) or gain <= 0:
        raise argparse.ArgumentTypeError("gain must be a finite positive number")
    return gain


def load_rgba(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.array(image.convert("RGBA"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read image '{path}': {exc}") from exc


def check_output_path(output: Path, sources: tuple[Path, ...]) -> None:
    for source in sources:
        same_path = output.resolve() == source.resolve()
        same_file = output.exists() and source.exists() and output.samefile(source)
        if same_path or same_file:
            raise ValueError(f"Output must not overwrite another image: '{source}'")


def save_png(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Save a visible, opaque RGBA difference PNG. RGB differences are "
            "amplified; alpha differences also brighten all three color channels. "
            "Black means identical. Images must have equal dimensions."
        ),
    )
    parser.add_argument("a", type=Path, help="original image")
    parser.add_argument("b", type=Path, help="processed image")
    parser.add_argument(
        "--out", type=Path, default=Path("out/diff.png"),
        help="visible opaque RGBA PNG path (default: out/diff.png)",
    )
    parser.add_argument(
        "--gain", type=positive_gain, default=8.0,
        help="difference brightness multiplier (default: 8)",
    )
    parser.add_argument(
        "--raw-out", type=Path,
        help="optional per-channel RGBA delta; use --gain 1 for exact values (may be transparent)",
    )
    parser.add_argument(
        "--preview-out", type=Path,
        help="optional opaque RGBA preview: RGB difference on the left, alpha on the right",
    )
    args = parser.parse_args()

    try:
        outputs = [path for path in (args.out, args.raw_out, args.preview_out) if path is not None]
        for index, output in enumerate(outputs):
            check_output_path(output, (args.a, args.b, *outputs[:index]))
        a, b = load_rgba(args.a), load_rgba(args.b)
        if a.shape != b.shape:
            raise ValueError(
                f"Image dimensions differ: '{args.a}' is {a.shape[1]}x{a.shape[0]}, "
                f"'{args.b}' is {b.shape[1]}x{b.shape[0]}. "
                "Use images with matching dimensions; no resizing is performed."
            )

        # Signed subtraction prevents uint8 wraparound. A gain >= 255 saturates
        # every nonzero channel, so capping it also avoids floating-point overflow.
        delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
        amplified = np.clip(delta * min(args.gain, 255.0), 0, 255).astype(np.uint8)
        # The display alpha must stay opaque; zero alpha difference means equal
        # transparency, not that the RGB differences should become invisible.
        visual = np.full_like(amplified, 255)
        visual[:, :, :3] = np.maximum(amplified[:, :, :3], amplified[:, :, 3:4])
        save_png(args.out, visual)
        if args.raw_out is not None:
            save_png(args.raw_out, amplified)
        if args.preview_out is not None:
            height, width = visual.shape[:2]
            preview = np.full((height, width * 2, 4), 255, dtype=np.uint8)
            preview[:, :width, :3] = amplified[:, :, :3]
            preview[:, width:, :3] = amplified[:, :, 3:4]
            save_png(args.preview_out, preview)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    changed = int(np.count_nonzero(np.any(delta != 0, axis=2)))
    total = delta.shape[0] * delta.shape[1]
    print(f"Saved visible RGBA diff (opaque; color and alpha changes included): {args.out.resolve()}")
    print(f"Size: {delta.shape[1]}x{delta.shape[0]}; gain: {args.gain:g}")
    print(f"Changed pixels: {changed}/{total} ({changed / total:.2%})")
    print(f"Mean absolute RGBA channel difference: {delta.mean():.3f}/255")
    print(f"Mean absolute alpha difference: {delta[:, :, 3].mean():.3f}/255")
    if args.raw_out is not None:
        print(f"Saved raw RGBA delta (may be transparent): {args.raw_out.resolve()}")
    if args.preview_out is not None:
        print(f"Saved opaque preview (left: RGB diff, right: alpha diff): {args.preview_out.resolve()}")


if __name__ == "__main__":
    main()
