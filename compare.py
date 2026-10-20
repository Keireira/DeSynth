# Quantitative similarity metrics between two images. No model downloads —
# pure pixel-level + structural metrics from numpy/skimage/opencv.

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).parent


def load(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"can't read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _ssim_rgb(a: np.ndarray, b: np.ndarray):
    if min(a.shape[:2]) < 7:
        return None
    # skimage's default SSIM uses a 7x7 window and excludes three border pixels.
    # Three-row halos make each strip equivalent to the corresponding region
    # of the full image; only scratch memory is reduced, never the resolution.
    total, rows = 0.0, a.shape[0] - 6
    for start in range(3, a.shape[0] - 3, 128):
        stop = min(start + 128, a.shape[0] - 3)
        for channel in range(3):
            aa = a[start - 3:stop + 3, :, channel].astype(np.float32)
            bb = b[start - 3:stop + 3, :, channel].astype(np.float32)
            total += float(ssim(aa, bb, data_range=255)) * (stop - start)
    return total / (rows * 3)


def basic_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact pixel metrics and SSIM for UI previews, with bounded scratch space."""
    if a.shape != b.shape or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("Metrics require RGB images with matching dimensions.")
    squared, absolute = 0.0, 0.0
    for row in range(0, a.shape[0], 128):
        delta = a[row:row + 128].astype(np.float32) - b[row:row + 128]
        absolute += float(np.abs(delta).sum(dtype=np.float64))
        squared += float(np.square(delta).sum(dtype=np.float64))
    mse = squared / a.size
    return dict(shape=a.shape, mse=mse, mae=absolute / a.size,
                psnr_db=float(10 * np.log10(255**2 / mse)) if mse else float("inf"),
                ssim=_ssim_rgb(a, b))


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """Extended, full-resolution analysis used by the comparison CLI."""
    result = basic_metrics(a, b)
    hist_corr, low_scores, high_scores = [], [], []
    for channel in range(3):
        aa, bb = a[..., channel].copy(), b[..., channel].copy()
        ha = cv2.calcHist([aa], [0], None, [256], [0, 256]).flatten()
        hb = cv2.calcHist([bb], [0], None, [256], [0, 256]).flatten()
        hist_corr.append(float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)))
        af, bf = aa.astype(np.float32), bb.astype(np.float32)
        low_a = cv2.GaussianBlur(af, (0, 0), sigmaX=1.95, sigmaY=1.95)
        low_b = cv2.GaussianBlur(bf, (0, 0), sigmaX=1.95, sigmaY=1.95)
        if min(aa.shape) >= 7:
            low_scores.append(ssim(low_a.clip(0, 255).astype(np.uint8).astype(np.float32),
                                   low_b.clip(0, 255).astype(np.uint8).astype(np.float32), data_range=255))
            high_a = (af - low_a + 128).clip(0, 255).astype(np.uint8).astype(np.float32)
            high_b = (bf - low_b + 128).clip(0, 255).astype(np.uint8).astype(np.float32)
            high_scores.append(ssim(high_a, high_b, data_range=255))
    result.update(ssim_low_pass=float(np.mean(low_scores)) if low_scores else None,
                  ssim_high_pass=float(np.mean(high_scores)) if high_scores else None,
                  hist_corr_rgb=hist_corr, mean_hist_corr=float(np.mean(hist_corr)))
    return result


def _draw_visual(a: np.ndarray, b: np.ndarray, m: dict, out_path: Path) -> None:
    # Side-by-side, normalized height, with a header strip carrying the stats.
    target_h = 720
    def fit(img):
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * target_h / h), target_h), interpolation=cv2.INTER_AREA)
    A, B = fit(a), fit(b)
    gap = 16
    label_h = 36
    stats_h = 84
    pad = 24
    canvas_w = A.shape[1] + B.shape[1] + gap + pad * 2
    canvas_h = label_h + target_h + stats_h + pad * 2
    canvas = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)

    # cv2 wants BGR; we've been working in RGB.
    A_bgr = cv2.cvtColor(A, cv2.COLOR_RGB2BGR)
    B_bgr = cv2.cvtColor(B, cv2.COLOR_RGB2BGR)
    y0 = pad + label_h
    canvas[y0:y0 + target_h, pad:pad + A.shape[1]] = A_bgr
    canvas[y0:y0 + target_h, pad + A.shape[1] + gap:pad + A.shape[1] + gap + B.shape[1]] = B_bgr

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "reference", (pad, pad + 24), font, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "desynth output", (pad + A.shape[1] + gap, pad + 24), font, 0.7, (220, 220, 220), 1, cv2.LINE_AA)

    stats_y = y0 + target_h + 28
    line1 = f"PSNR {m['psnr_db']:.2f} dB   SSIM {m['ssim']:.4f}   MAE {m['mae']:.2f}"
    line2 = f"SSIM low {m['ssim_low_pass']:.4f}   SSIM high {m['ssim_high_pass']:.4f}   hist corr {m['mean_hist_corr']:.4f}"
    cv2.putText(canvas, line1, (pad, stats_y), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, line2, (pad, stats_y + 28), font, 0.6, (180, 200, 220), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)
    print(f"  -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a", help="reference image")
    parser.add_argument("b", help="comparison image")
    parser.add_argument("--visual", action="store_true", help="save side-by-side PNG")
    parser.add_argument("--visual-out", type=Path, default=None, help="output path for --visual")
    args = parser.parse_args()

    a = load(Path(args.a))
    b = load(Path(args.b))
    if a.shape != b.shape:
        print(f"resizing b from {b.shape} to {a.shape}")
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    m = metrics(a, b)
    print(f"  shape:           {m['shape']}")
    print(f"  PSNR:            {m['psnr_db']:.2f} dB        (higher = closer; >40 typically imperceptible)")
    print(f"  SSIM:            {m['ssim']:.4f}            (1.0 = identical)")
    print(f"   - low-pass:     {m['ssim_low_pass']:.4f}            (broad composition agreement)")
    print(f"   - high-pass:    {m['ssim_high_pass']:.4f}            (fine-detail agreement)")
    print(f"  MAE:             {m['mae']:.3f}             (mean abs pixel diff, 0-255 scale)")
    print(f"  MSE:             {m['mse']:.3f}")
    print(f"  hist correl R/G/B: {m['hist_corr_rgb'][0]:.4f} / {m['hist_corr_rgb'][1]:.4f} / {m['hist_corr_rgb'][2]:.4f}")

    if args.visual:
        out = args.visual_out or (ROOT / "out" / f"comparison_{Path(args.b).stem}.png")
        out.parent.mkdir(exist_ok=True)
        _draw_visual(a, b, m, out)


if __name__ == "__main__":
    main()
