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

    parser = argparse.ArgumentParser(
        description="Draw intermediate images for Laplacian sharpness analysis."
    )
    parser.add_argument(
        "--current",
        default="650",
        help="Condenser current tag in the image filename.",
    )
    parser.add_argument(
        "--magnification",
        default="11k",
        help="Magnification tag in the image filename.",
    )
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
        help="Pixels removed from the image bottom before Laplacian analysis.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=sem_dir / "anly_data" / "sharpness_process",
        help="Directory for the process figure.",
    )
    return parser.parse_args()


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image_min = float(image.min())
    image_max = float(image.max())
    if image_max == image_min:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - image_min) / (image_max - image_min)
    return np.clip(scaled * 255, 0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    image_path = args.image_dir / f"{args.current}_{args.magnification}.png"
    gray_full = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray_full is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    if args.crop_bottom < 0:
        raise ValueError("--crop-bottom must be non-negative.")
    if args.crop_bottom >= gray_full.shape[0]:
        raise ValueError(
            f"--crop-bottom={args.crop_bottom} is too large for image height {gray_full.shape[0]}."
        )

    if args.crop_bottom:
        gray_roi = gray_full[:-args.crop_bottom, :]
        crop_line_y = gray_roi.shape[0]
    else:
        gray_roi = gray_full
        crop_line_y = None

    laplacian = cv2.Laplacian(gray_roi, cv2.CV_64F, ksize=3)
    abs_laplacian = np.abs(laplacian)
    abs_laplacian_u8 = normalize_to_uint8(abs_laplacian)
    sharpness = float(laplacian.var(ddof=0))

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.4), dpi=160, constrained_layout=True)

    ax = axes[0, 0]
    ax.imshow(gray_full, cmap="gray", vmin=0, vmax=255)
    if crop_line_y is not None:
        ax.axhline(crop_line_y, color="yellow", linewidth=1.6)
    ax.set_title("Original grayscale image")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(gray_roi, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Analysis ROI")
    ax.axis("off")

    ax = axes[1, 0]
    limit = float(np.percentile(np.abs(laplacian), 99))
    if limit == 0:
        limit = 1.0
    lap_img = ax.imshow(laplacian, cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_title("Signed Laplacian response")
    ax.axis("off")
    fig.colorbar(lap_img, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    ax.imshow(abs_laplacian_u8, cmap="magma", vmin=0, vmax=255)
    ax.set_title("Abs. Laplacian strength")
    ax.axis("off")

    fig.suptitle(
        f"Sharpness process: {image_path.name}, Var(Laplacian) = {sharpness:.3f}",
        fontsize=14,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"sharpness_process_{args.current}_{args.magnification}.png"
    fig.savefig(out_path)
    plt.close(fig)

    print(f"image: {image_path}")
    print(f"roi: width={gray_roi.shape[1]}, height={gray_roi.shape[0]}")
    print(f"mean_gray: {gray_roi.mean():.6f}")
    print(f"laplacian_std: {laplacian.std(ddof=0):.6f}")
    print(f"sharpness_var_laplacian: {sharpness:.6f}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
