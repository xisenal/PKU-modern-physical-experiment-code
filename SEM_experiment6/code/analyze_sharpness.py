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
class SharpnessRecord:
    current: float
    magnification: str
    filename: str
    width: int
    height: int
    roi: str
    mean_gray: float
    laplacian_mean: float
    laplacian_std: float
    sharpness_var_laplacian: float


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    sem_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Analyze SEM sharpness using the variance of the Laplacian."
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
        "--out-dir",
        type=Path,
        default=sem_dir / "anly_data" / "sharpness",
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

    return gray, roi


def analyze_images(
    image_dir: Path,
    magnification: str,
    crop_bottom: int,
) -> list[SharpnessRecord]:
    image_files = iter_image_files(image_dir, magnification)
    if not image_files:
        raise FileNotFoundError(
            f"No images matching '*_{magnification}.*' were found in {image_dir}."
        )

    records: list[SharpnessRecord] = []
    for path, current, mag in image_files:
        gray, roi = read_gray_roi(path, crop_bottom)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        records.append(
            SharpnessRecord(
                current=current,
                magnification=mag,
                filename=path.name,
                width=int(gray.shape[1]),
                height=int(gray.shape[0]),
                roi=roi,
                mean_gray=float(gray.mean()),
                laplacian_mean=float(laplacian.mean()),
                laplacian_std=float(laplacian.std(ddof=0)),
                sharpness_var_laplacian=float(laplacian.var(ddof=0)),
            )
        )

    return records


def write_csv(records: list[SharpnessRecord], csv_path: Path) -> None:
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
                "laplacian_mean",
                "laplacian_std",
                "sharpness_var_laplacian",
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
                    f"{record.laplacian_mean:.6f}",
                    f"{record.laplacian_std:.6f}",
                    f"{record.sharpness_var_laplacian:.6f}",
                ]
            )


def write_markdown(records: list[SharpnessRecord], md_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| current | magnification | file | mean gray | Laplacian std | sharpness Var(Laplacian) |",
        "|---:|:---:|:---|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            "| "
            f"{record.current:g} | "
            f"{record.magnification} | "
            f"{record.filename} | "
            f"{record.mean_gray:.3f} | "
            f"{record.laplacian_std:.3f} | "
            f"{record.sharpness_var_laplacian:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_sharpness(records: list[SharpnessRecord], plot_path: Path) -> None:
    currents = [record.current for record in records]
    sharpness_values = [record.sharpness_var_laplacian for record in records]

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=160)
    ax.plot(currents, sharpness_values, marker="o", linewidth=1.8)
    ax.set_xlabel("Condenser current")
    ax.set_ylabel("Sharpness Var(Laplacian)")
    ax.set_title(f"SEM sharpness vs condenser current ({records[0].magnification})")
    ax.grid(True, alpha=0.28)
    ax.set_axisbelow(True)
    ax.margins(y=0.12)

    best = max(records, key=lambda record: record.sharpness_var_laplacian)
    ax.scatter([best.current], [best.sharpness_var_laplacian], s=70, zorder=3)
    ax.annotate(
        f"max: {best.current:g}",
        xy=(best.current, best.sharpness_var_laplacian),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)


def print_table(records: list[SharpnessRecord]) -> None:
    print("current  magnification  mean_gray  lap_std  sharpness_var_laplacian")
    for record in records:
        print(
            f"{record.current:7g}  {record.magnification:>13}  "
            f"{record.mean_gray:9.3f}  {record.laplacian_std:7.3f}  "
            f"{record.sharpness_var_laplacian:24.3f}"
        )


def main() -> None:
    args = parse_args()
    records = analyze_images(args.image_dir, args.magnification, args.crop_bottom)

    suffix = args.magnification.lower().replace(".", "p")
    csv_path = args.out_dir / f"sharpness_data_{suffix}.csv"
    md_path = args.out_dir / f"sharpness_data_{suffix}.md"
    plot_path = args.out_dir / f"sharpness_vs_current_{suffix}.png"

    write_csv(records, csv_path)
    write_markdown(records, md_path)
    plot_sharpness(records, plot_path)
    print_table(records)
    print()
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()
