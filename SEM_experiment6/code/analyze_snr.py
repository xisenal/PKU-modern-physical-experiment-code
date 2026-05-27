from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import cv2
import matplotlib.pyplot as plt
import numpy as np


FILENAME_RE = re.compile(
    r"^(?P<current>\d+(?:\.\d+)?)_(?P<magnification>\d+(?:\.\d+)?k)\.(?P<ext>png|jpg|jpeg|tif|tiff)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SNRRecord:
    current: float
    magnification: str
    filename: str
    width: int
    height: int
    roi: str
    mean_gray: float
    noise_sigma_flat: float
    snr_flat: float
    noise_sigma_all: float
    snr_all: float
    flat_pixel_fraction: float


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    sem_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Analyze SEM SNR using high-pass residual noise estimation."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=sem_dir / "pic",
        help="Directory containing SEM images named like 400_11k.png.",
    )
    parser.add_argument(
        "--magnification",
        default="11k",
        help="Magnification tag to analyze, matching the second number in each filename.",
    )
    parser.add_argument(
        "--crop-bottom",
        type=int,
        default=80,
        help=(
            "Pixels removed from the image bottom before analysis. "
            "Use 0 to include the full frame."
        ),
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
        help=(
            "Percentile threshold of the smoothed-image Sobel gradient. "
            "Pixels below this threshold are treated as relatively flat."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=sem_dir / "anly_data" / "snr",
        help="Directory for the output table and plot.",
    )
    return parser.parse_args()


def iter_image_files(image_dir: Path, magnification: str) -> list[tuple[Path, float, str]]:
    files: list[tuple[Path, float, str]] = []
    wanted_mag = magnification.lower()

    for path in image_dir.iterdir():
        if not path.is_file():
            continue
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        mag = match.group("magnification").lower()
        if mag != wanted_mag:
            continue
        files.append((path, float(match.group("current")), mag))

    return sorted(files, key=lambda item: item[1])


def read_gray_roi(path: Path, crop_bottom: int) -> tuple[np.ndarray, str]:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Failed to read image: {path}")

    if crop_bottom < 0:
        raise ValueError("--crop-bottom must be non-negative.")
    if crop_bottom >= gray.shape[0]:
        raise ValueError(
            f"--crop-bottom={crop_bottom} is too large for {path.name} "
            f"with height {gray.shape[0]}."
        )
    if crop_bottom:
        gray = gray[:-crop_bottom, :]
        roi = f"x=0:{gray.shape[1]}, y=0:{gray.shape[0]} (bottom {crop_bottom}px cropped)"
    else:
        roi = f"x=0:{gray.shape[1]}, y=0:{gray.shape[0]} (full image)"

    return gray.astype(np.float64), roi


def estimate_snr(
    gray: np.ndarray,
    gaussian_sigma: float,
    flat_percentile: float,
) -> tuple[float, float, float, float, float]:
    if gaussian_sigma <= 0:
        raise ValueError("--gaussian-sigma must be positive.")
    if not 0 < flat_percentile <= 100:
        raise ValueError("--flat-percentile must be in (0, 100].")

    smooth = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=gaussian_sigma, sigmaY=gaussian_sigma)
    highpass = gray - smooth

    grad_x = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    threshold = float(np.percentile(gradient, flat_percentile))
    flat_mask = gradient <= threshold

    noise_sigma_all = float(highpass.std(ddof=0))
    noise_sigma_flat = float(highpass[flat_mask].std(ddof=0))
    mean_gray = float(gray.mean())
    snr_all = mean_gray / noise_sigma_all if noise_sigma_all else float("inf")
    snr_flat = mean_gray / noise_sigma_flat if noise_sigma_flat else float("inf")
    flat_pixel_fraction = float(flat_mask.mean())
    return noise_sigma_flat, snr_flat, noise_sigma_all, snr_all, flat_pixel_fraction


def analyze_images(
    image_dir: Path,
    magnification: str,
    crop_bottom: int,
    gaussian_sigma: float,
    flat_percentile: float,
) -> list[SNRRecord]:
    image_files = iter_image_files(image_dir, magnification)
    if not image_files:
        raise FileNotFoundError(
            f"No images matching '*_{magnification}.*' were found in {image_dir}."
        )

    records: list[SNRRecord] = []
    for path, current, mag in image_files:
        gray, roi = read_gray_roi(path, crop_bottom)
        noise_sigma_flat, snr_flat, noise_sigma_all, snr_all, flat_pixel_fraction = estimate_snr(
            gray, gaussian_sigma, flat_percentile
        )
        records.append(
            SNRRecord(
                current=current,
                magnification=mag,
                filename=path.name,
                width=int(gray.shape[1]),
                height=int(gray.shape[0]),
                roi=roi,
                mean_gray=float(gray.mean()),
                noise_sigma_flat=noise_sigma_flat,
                snr_flat=snr_flat,
                noise_sigma_all=noise_sigma_all,
                snr_all=snr_all,
                flat_pixel_fraction=flat_pixel_fraction,
            )
        )

    return records


def write_csv(records: list[SNRRecord], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "current",
                "magnification",
                "filename",
                "width",
                "height",
                "roi",
                "mean_gray",
                "noise_sigma_flat",
                "snr_flat",
                "noise_sigma_all",
                "snr_all",
                "flat_pixel_fraction",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    f"{record.current:g}",
                    record.magnification,
                    record.filename,
                    record.width,
                    record.height,
                    record.roi,
                    f"{record.mean_gray:.6f}",
                    f"{record.noise_sigma_flat:.6f}",
                    f"{record.snr_flat:.6f}",
                    f"{record.noise_sigma_all:.6f}",
                    f"{record.snr_all:.6f}",
                    f"{record.flat_pixel_fraction:.6f}",
                ]
            )


def write_markdown(records: list[SNRRecord], md_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| current | magnification | file | mean gray | noise sigma flat | SNR flat | noise sigma all | SNR all | flat fraction |",
        "|---:|:---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            "| "
            f"{record.current:g} | "
            f"{record.magnification} | "
            f"{record.filename} | "
            f"{record.mean_gray:.3f} | "
            f"{record.noise_sigma_flat:.3f} | "
            f"{record.snr_flat:.3f} | "
            f"{record.noise_sigma_all:.3f} | "
            f"{record.snr_all:.3f} | "
            f"{record.flat_pixel_fraction:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_snr(records: list[SNRRecord], plot_path: Path) -> None:
    currents = [record.current for record in records]
    snr_flat_values = [record.snr_flat for record in records]
    snr_all_values = [record.snr_all for record in records]

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=160)
    ax.plot(currents, snr_flat_values, marker="o", linewidth=1.8, label="flat-region SNR")
    ax.plot(currents, snr_all_values, marker="s", linewidth=1.4, alpha=0.72, label="whole-ROI SNR")
    ax.set_xlabel("Condenser current")
    ax.set_ylabel("SNR = mean gray / noise sigma")
    ax.set_title(f"SEM SNR vs condenser current ({records[0].magnification})")
    ax.grid(True, alpha=0.28)
    ax.set_axisbelow(True)
    ax.margins(y=0.12)
    ax.legend(frameon=False)

    best = max(records, key=lambda record: record.snr_flat)
    ax.scatter([best.current], [best.snr_flat], s=70, zorder=3)
    ax.annotate(
        f"max: {best.current:g}",
        xy=(best.current, best.snr_flat),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)


def print_table(records: list[SNRRecord]) -> None:
    print("current  magnification  mean_gray  noise_sigma_flat  snr_flat  noise_sigma_all  snr_all")
    for record in records:
        print(
            f"{record.current:7g}  {record.magnification:>13}  "
            f"{record.mean_gray:9.3f}  {record.noise_sigma_flat:16.3f}  "
            f"{record.snr_flat:8.3f}  {record.noise_sigma_all:15.3f}  "
            f"{record.snr_all:7.3f}"
        )


def main() -> None:
    args = parse_args()
    records = analyze_images(
        args.image_dir,
        args.magnification,
        args.crop_bottom,
        args.gaussian_sigma,
        args.flat_percentile,
    )

    suffix = args.magnification.lower().replace(".", "p")
    csv_path = args.out_dir / f"snr_data_{suffix}.csv"
    md_path = args.out_dir / f"snr_data_{suffix}.md"
    plot_path = args.out_dir / f"snr_vs_current_{suffix}.png"

    write_csv(records, csv_path)
    write_markdown(records, md_path)
    plot_snr(records, plot_path)
    print_table(records)
    print()
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()
