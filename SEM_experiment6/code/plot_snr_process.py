from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    sem_dir = script_dir.parent

    parser = argparse.ArgumentParser(description="Draw intermediate images for SEM SNR analysis.")
    parser.add_argument("--current", default="650", help="Condenser current tag in the image filename.")
    parser.add_argument("--magnification", default="11k", help="Magnification tag in the image filename.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=sem_dir / "pic",
        help="Directory containing SEM images.",
    )
    parser.add_argument(
        "--crop-bottom",
        type=int,
        default=80,
        help="Pixels removed from the image bottom before SNR analysis.",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=3.0,
        help="Gaussian blur sigma used to estimate the low-frequency image background.",
    )
    parser.add_argument(
        "--flat-percentile",
        type=float,
        default=50.0,
        help="Gradient percentile used to select relatively flat pixels.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=sem_dir / "anly_data" / "snr_process",
        help="Directory for the process figure.",
    )
    return parser.parse_args()


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    image_min = float(image.min())
    image_max = float(image.max())
    if image_max == image_min:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - image_min) / (image_max - image_min)
    return np.clip(scaled * 255, 0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    image_path = args.image_dir / f"{args.current}_{args.magnification}.png"
    gray_full_u8 = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray_full_u8 is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    if args.crop_bottom < 0:
        raise ValueError("--crop-bottom must be non-negative.")
    if args.crop_bottom >= gray_full_u8.shape[0]:
        raise ValueError(
            f"--crop-bottom={args.crop_bottom} is too large for image height {gray_full_u8.shape[0]}."
        )
    if args.gaussian_sigma <= 0:
        raise ValueError("--gaussian-sigma must be positive.")
    if not 0 < args.flat_percentile <= 100:
        raise ValueError("--flat-percentile must be in (0, 100].")

    if args.crop_bottom:
        gray_roi = gray_full_u8[:-args.crop_bottom, :].astype(np.float64)
        crop_line_y = gray_roi.shape[0]
    else:
        gray_roi = gray_full_u8.astype(np.float64)
        crop_line_y = None

    smooth = cv2.GaussianBlur(
        gray_roi,
        ksize=(0, 0),
        sigmaX=args.gaussian_sigma,
        sigmaY=args.gaussian_sigma,
    )
    highpass = gray_roi - smooth

    grad_x = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    flat_threshold = float(np.percentile(gradient, args.flat_percentile))
    flat_mask = gradient <= flat_threshold

    mean_gray = float(gray_roi.mean())
    noise_sigma_flat = float(highpass[flat_mask].std(ddof=0))
    noise_sigma_all = float(highpass.std(ddof=0))
    snr_flat = mean_gray / noise_sigma_flat if noise_sigma_flat else float("inf")
    snr_all = mean_gray / noise_sigma_all if noise_sigma_all else float("inf")

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), dpi=160, constrained_layout=True)

    ax = axes[0, 0]
    ax.imshow(gray_full_u8, cmap="gray", vmin=0, vmax=255)
    if crop_line_y is not None:
        ax.axhline(crop_line_y, color="yellow", linewidth=1.6)
    ax.set_title("Original grayscale image")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(gray_roi, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Analysis ROI")
    ax.axis("off")

    ax = axes[0, 2]
    ax.imshow(smooth, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"Gaussian background, sigma={args.gaussian_sigma:g}")
    ax.axis("off")

    ax = axes[1, 0]
    residual_limit = float(np.percentile(np.abs(highpass), 99))
    if residual_limit == 0:
        residual_limit = 1.0
    residual_img = ax.imshow(highpass, cmap="coolwarm", vmin=-residual_limit, vmax=residual_limit)
    ax.set_title("High-pass residual")
    ax.axis("off")
    fig.colorbar(residual_img, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    mask_display = normalize_for_display(gradient)
    ax.imshow(mask_display, cmap="gray")
    ax.imshow(np.ma.masked_where(~flat_mask, flat_mask), cmap="autumn", alpha=0.45)
    ax.set_title(f"Flat-region mask, lowest {args.flat_percentile:g}% gradient")
    ax.axis("off")

    ax = axes[1, 2]
    flat_noise = highpass[flat_mask].ravel()
    ax.hist(flat_noise, bins=80, color="#2f6f9f", alpha=0.82)
    ax.axvline(noise_sigma_flat, color="#c43d3d", linewidth=1.4, label="+1 sigma")
    ax.axvline(-noise_sigma_flat, color="#c43d3d", linewidth=1.4, label="-1 sigma")
    ax.set_title("Flat-region residual histogram")
    ax.set_xlabel("Residual gray value")
    ax.set_ylabel("Pixel count")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        (
            f"SNR process: {image_path.name}, "
            f"SNR(flat) = {snr_flat:.3f}, SNR(all) = {snr_all:.3f}"
        ),
        fontsize=14,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"snr_process_{args.current}_{args.magnification}.png"
    fig.savefig(out_path)
    plt.close(fig)

    print(f"image: {image_path}")
    print(f"roi: width={gray_roi.shape[1]}, height={gray_roi.shape[0]}")
    print(f"mean_gray: {mean_gray:.6f}")
    print(f"noise_sigma_flat: {noise_sigma_flat:.6f}")
    print(f"snr_flat: {snr_flat:.6f}")
    print(f"noise_sigma_all: {noise_sigma_all:.6f}")
    print(f"snr_all: {snr_all:.6f}")
    print(f"flat_pixel_fraction: {flat_mask.mean():.6f}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
