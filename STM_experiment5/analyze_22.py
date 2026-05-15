#!/usr/bin/env python3
"""
Analyze STM/22.png with the same atom-spacing method used for 11-1.png.

Output:
    STM/22_annotated.png
    printed x/y scan lengths in nm
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analyze_11_1 import (
    ATOM_SPACING_NM,
    annotate,
    detect_atom_centers,
    estimate_lattice_vectors,
    estimate_scan_size,
    grayscale,
    read_png_rgba,
    write_png_rgba,
)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Estimate STM x/y scan length from atom spacings.")
    parser.add_argument("image", nargs="?", default=str(script_dir / "22.png"), help="input PNG image")
    parser.add_argument("-o", "--output", default=None, help="annotated output PNG")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    out_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else image_path.with_name(image_path.stem + "_annotated.png")
    )

    width, height, pixels = read_png_rgba(image_path)
    centers = detect_atom_centers(grayscale(pixels))
    hvec, dvec = estimate_lattice_vectors(centers)
    metrics = estimate_scan_size(width, height, hvec, dvec)

    annotated = annotate(pixels, centers, hvec, dvec, metrics)
    write_png_rgba(out_path, width, height, annotated)

    print(f"input: {image_path}")
    print(f"annotated image: {out_path}")
    print(f"detected atoms: {len(centers)}")
    print(f"horizontal atom spacing: {metrics['x_pixels_per_atom']:.2f} px = {ATOM_SPACING_NM:.3f} nm")
    print(f"diagonal atom vector: dx={dvec[0]:.2f} px, dy={dvec[1]:.2f} px")
    print(f"x axis: {metrics['x_atom_spacings']:.2f} atom spacings, total length = {metrics['x_length_nm']:.3f} nm")
    print(f"y axis: {metrics['y_atom_spacings']:.2f} atom spacings, total length = {metrics['y_length_nm']:.3f} nm")


if __name__ == "__main__":
    main()
