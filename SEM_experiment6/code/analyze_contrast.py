from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



FILENAME_RE = re.compile(
    r"^(?P<current>\d+(?:\.\d+)?)_(?P<magnification>\d+(?:\.\d+)?k)\.(?P<ext>png|jpg|jpeg|tif|tiff)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ContrastRecord:
    current: float
    magnification: str
    filename: str
    width: int
    height: int
    roi: str
    mean_gray: float
    std_gray: float
    contrast: float


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    sem_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Analyze SEM image contrast as a function of condenser current."
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
        default=sem_dir,
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
    image = Image.open(path).convert("L")
    gray = np.asarray(image, dtype=np.float64)

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
) -> list[ContrastRecord]:
    image_files = iter_image_files(image_dir, magnification)
    if not image_files:
        raise FileNotFoundError(
            f"No images matching '*_{magnification}.*' were found in {image_dir}."
        )

    records: list[ContrastRecord] = []
    for path, current, mag in image_files:
        gray, roi = read_gray_roi(path, crop_bottom)
        mean_gray = float(gray.mean())
        std_gray = float(gray.std(ddof=0))
        contrast = std_gray / mean_gray if mean_gray else float("nan")
        records.append(
            ContrastRecord(
                current=current,
                magnification=mag,
                filename=path.name,
                width=int(gray.shape[1]),
                height=int(gray.shape[0]),
                roi=roi,
                mean_gray=mean_gray,
                std_gray=std_gray,
                contrast=contrast,
            )
        )

    return records


def write_csv(records: list[ContrastRecord], csv_path: Path) -> None:
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
                "std_gray",
                "contrast_std_over_mean",
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
                    f"{record.std_gray:.6f}",
                    f"{record.contrast:.6f}",
                ]
            )


def write_markdown(records: list[ContrastRecord], md_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| current | magnification | file | mean gray | std gray | contrast std/mean |",
        "|---:|:---:|:---|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            "| "
            f"{record.current:g} | "
            f"{record.magnification} | "
            f"{record.filename} | "
            f"{record.mean_gray:.3f} | "
            f"{record.std_gray:.3f} | "
            f"{record.contrast:.6f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_contrast(records: list[ContrastRecord], plot_path: Path) -> None:
    currents = [record.current for record in records]
    contrasts = [record.contrast for record in records]

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=160)
    ax.plot(currents, contrasts, marker="o", linewidth=1.8)
    ax.set_xlabel("Condenser current")
    ax.set_ylabel("Contrast (gray std / gray mean)")
    ax.set_title(f"SEM contrast vs condenser current ({records[0].magnification})")
    ax.grid(True, alpha=0.28)
    ax.set_axisbelow(True)
    ax.margins(y=0.12)

    best = max(records, key=lambda record: record.contrast)
    ax.scatter([best.current], [best.contrast], s=70, zorder=3)
    ax.annotate(
        f"max: {best.current:g}",
        xy=(best.current, best.contrast),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)


def print_table(records: list[ContrastRecord]) -> None:
    print("current  magnification  mean_gray  std_gray  contrast")
    for record in records:
        print(
            f"{record.current:7g}  {record.magnification:>13}  "
            f"{record.mean_gray:9.3f}  {record.std_gray:8.3f}  "
            f"{record.contrast:8.6f}"
        )


def main() -> None:
    args = parse_args()
    records = analyze_images(args.image_dir, args.magnification, args.crop_bottom)

    suffix = args.magnification.lower().replace(".", "p")
    csv_path = args.out_dir / f"contrast_data_{suffix}.csv"
    md_path = args.out_dir / f"contrast_data_{suffix}.md"
    plot_path = args.out_dir / f"contrast_vs_current_{suffix}.png"

    write_csv(records, csv_path)
    write_markdown(records, md_path)
    plot_contrast(records, plot_path)
    print_table(records)
    print()
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()
