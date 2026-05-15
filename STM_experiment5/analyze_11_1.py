#!/usr/bin/env python3
"""
Estimate the STM scan size from atomic-resolution graphite/graphene dots.

The script is intentionally dependency-free: it uses only the Python standard
library, because many lab machines do not have numpy/Pillow/opencv installed.

Default input:
    STM/11-1.png

Output:
    STM/11-1_annotated.png
    printed x/y scan lengths in nm
"""

from __future__ import annotations

import argparse
import math
import statistics
import struct
import zlib
from pathlib import Path


ATOM_SPACING_NM = 0.246


FONT_5X7 = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11110", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "=": ["00000", "11110", "00000", "11110", "00000", "00000", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "11100"],
    "a": ["00000", "01110", "00001", "01111", "10001", "10011", "01101"],
    "d": ["00001", "00001", "01101", "10011", "10001", "10011", "01101"],
    "e": ["00000", "01110", "10001", "11111", "10000", "10001", "01110"],
    "g": ["00000", "01101", "10011", "10001", "01111", "00001", "01110"],
    "i": ["00100", "00000", "01100", "00100", "00100", "00100", "01110"],
    "m": ["00000", "11010", "10101", "10101", "10101", "10101", "10101"],
    "n": ["00000", "10110", "11001", "10001", "10001", "10001", "10001"],
    "o": ["00000", "01110", "10001", "10001", "10001", "10001", "01110"],
    "p": ["00000", "11110", "10001", "11110", "10000", "10000", "10000"],
    "s": ["00000", "01111", "10000", "01110", "00001", "00001", "11110"],
    "t": ["01000", "01000", "11110", "01000", "01000", "01001", "00110"],
    "x": ["00000", "10001", "01010", "00100", "01010", "10001", "00000"],
    "y": ["00000", "10001", "10001", "01111", "00001", "00010", "11100"],
}


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_png_rgba(path: Path) -> tuple[int, int, list[list[tuple[int, int, int, int]]]]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG file")

    width = height = bit_depth = color_type = None
    compressed = b""
    pos = 8
    while pos < len(data):
        size = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + size]
        pos += 12 + size
        if kind == b"IHDR":
            width, height, bit_depth, color_type, *_ = struct.unpack(">IIBBBBB", chunk)
        elif kind == b"IDAT":
            compressed += chunk
        elif kind == b"IEND":
            break

    if width is None or height is None:
        raise ValueError("PNG is missing IHDR")
    if bit_depth != 8 or color_type not in (0, 2, 6):
        raise ValueError("Only 8-bit grayscale/RGB/RGBA PNG files are supported")

    channels = {0: 1, 2: 3, 6: 4}[color_type]
    bpp = channels
    stride = width * channels
    raw = zlib.decompress(compressed)

    rows: list[list[tuple[int, int, int, int]]] = []
    prev = [0] * stride
    index = 0
    for _ in range(height):
        filter_type = raw[index]
        index += 1
        scan = list(raw[index : index + stride])
        index += stride

        recon = [0] * stride
        for i, byte in enumerate(scan):
            left = recon[i - bpp] if i >= bpp else 0
            up = prev[i]
            up_left = prev[i - bpp] if i >= bpp else 0
            if filter_type == 0:
                value = byte
            elif filter_type == 1:
                value = byte + left
            elif filter_type == 2:
                value = byte + up
            elif filter_type == 3:
                value = byte + ((left + up) // 2)
            elif filter_type == 4:
                value = byte + paeth(left, up, up_left)
            else:
                raise ValueError(f"Unsupported PNG filter: {filter_type}")
            recon[i] = value & 255

        row = []
        for x in range(width):
            if color_type == 0:
                g = recon[x]
                row.append((g, g, g, 255))
            elif color_type == 2:
                j = x * 3
                row.append((recon[j], recon[j + 1], recon[j + 2], 255))
            else:
                j = x * 4
                row.append((recon[j], recon[j + 1], recon[j + 2], recon[j + 3]))
        rows.append(row)
        prev = recon

    return width, height, rows


def write_png_rgba(path: Path, width: int, height: int, pixels: list[list[tuple[int, int, int, int]]]) -> None:
    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for r, g, b, a in row:
            raw.extend((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)), max(0, min(255, a))))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
        + png_chunk(b"IEND", b"")
    )
    path.write_bytes(data)


def grayscale(pixels: list[list[tuple[int, int, int, int]]]) -> list[list[float]]:
    return [[0.299 * r + 0.587 * g + 0.114 * b for r, g, b, _ in row] for row in pixels]


def row_detrend(gray: list[list[float]]) -> list[list[float]]:
    corrected = []
    for row in gray:
        med = statistics.median(row)
        corrected.append([v - med for v in row])
    return corrected


def box_blur(values: list[list[float]], radius: int) -> list[list[float]]:
    height, width = len(values), len(values[0])
    integral = [[0.0] * (width + 1) for _ in range(height + 1)]
    for y in range(height):
        running = 0.0
        prev_int = integral[y]
        cur_int = integral[y + 1]
        for x in range(width):
            running += values[y][x]
            cur_int[x + 1] = prev_int[x + 1] + running

    out = [[0.0] * width for _ in range(height)]
    for y in range(height):
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        for x in range(width):
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            area = (y1 - y0) * (x1 - x0)
            out[y][x] = (
                integral[y1][x1]
                - integral[y0][x1]
                - integral[y1][x0]
                + integral[y0][x0]
            ) / area
    return out


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(q * (len(ordered) - 1))))
    return ordered[index]


def detect_atom_centers(gray: list[list[float]]) -> list[tuple[float, float, float]]:
    corrected = row_detrend(gray)
    smooth = box_blur(corrected, radius=8)
    height, width = len(smooth), len(smooth[0])
    threshold = percentile([v for row in smooth for v in row], 0.93)

    local_maxima = []
    neighborhood = 15
    margin = 8
    for y in range(margin, height - margin):
        for x in range(margin, width - margin):
            value = smooth[y][x]
            if value < threshold:
                continue
            is_max = True
            for yy in range(max(0, y - neighborhood), min(height, y + neighborhood + 1)):
                if not is_max:
                    break
                for xx in range(max(0, x - neighborhood), min(width, x + neighborhood + 1)):
                    if smooth[yy][xx] > value:
                        is_max = False
                        break
            if is_max:
                local_maxima.append((x, y, value))

    local_maxima.sort(key=lambda p: p[2], reverse=True)
    centers: list[tuple[float, float, float]] = []
    min_separation = 38.0
    for x, y, value in local_maxima:
        if all(math.hypot(x - cx, y - cy) >= min_separation for cx, cy, _ in centers):
            centers.append((float(x), float(y), float(value)))
        if len(centers) >= 20:
            break

    centers.sort(key=lambda p: (p[1], p[0]))
    if len(centers) < 5:
        raise RuntimeError("Too few atom centers were detected; try adjusting threshold/radius in the script.")
    return centers


def median_vector(vectors: list[tuple[float, float]]) -> tuple[float, float]:
    return statistics.median([v[0] for v in vectors]), statistics.median([v[1] for v in vectors])


def estimate_lattice_vectors(centers: list[tuple[float, float, float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    pairs = []
    for i, (x1, y1, _) in enumerate(centers):
        for x2, y2, _ in centers[i + 1 :]:
            dx, dy = x2 - x1, y2 - y1
            dist = math.hypot(dx, dy)
            if 35 <= dist <= 95:
                pairs.append((dx, dy, dist))

    horizontal = [
        (dx, dy)
        for dx, dy, _ in pairs
        if dx > 45 and abs(dy) <= max(14, 0.25 * abs(dx))
    ]
    diagonal = [
        (dx, dy)
        for dx, dy, _ in pairs
        if dy > 25 and dx < -8 and abs(dx) < 65
    ]

    if len(horizontal) < 2:
        raise RuntimeError("Could not estimate the nearly-horizontal atom spacing.")
    if len(diagonal) < 2:
        raise RuntimeError("Could not estimate the diagonal atom spacing.")

    hvec = median_vector(horizontal)
    dvec = median_vector(diagonal)
    return hvec, dvec


def estimate_scan_size(
    width: int,
    height: int,
    hvec: tuple[float, float],
    dvec: tuple[float, float],
) -> dict[str, float]:
    hdx, _ = hvec
    ddx, ddy = dvec

    x_pixels_per_atom = abs(hdx)
    sx_nm_per_px = ATOM_SPACING_NM / x_pixels_per_atom

    x_part_nm = abs(ddx) * sx_nm_per_px
    if x_part_nm >= ATOM_SPACING_NM:
        # Conservative fallback: use the vertical projection of a triangular lattice.
        sy_nm_per_px = (ATOM_SPACING_NM * math.sqrt(3) / 2) / abs(ddy)
    else:
        sy_nm_per_px = math.sqrt(ATOM_SPACING_NM**2 - x_part_nm**2) / abs(ddy)

    x_length_nm = width * sx_nm_per_px
    y_length_nm = height * sy_nm_per_px
    return {
        "x_pixels_per_atom": x_pixels_per_atom,
        "y_pixels_per_atom_equiv": ATOM_SPACING_NM / sy_nm_per_px,
        "sx_nm_per_px": sx_nm_per_px,
        "sy_nm_per_px": sy_nm_per_px,
        "x_atom_spacings": x_length_nm / ATOM_SPACING_NM,
        "y_atom_spacings": y_length_nm / ATOM_SPACING_NM,
        "x_length_nm": x_length_nm,
        "y_length_nm": y_length_nm,
    }


def blend(dst: tuple[int, int, int, int], color: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r, g, b, a = dst
    cr, cg, cb, ca = color
    alpha = ca / 255.0
    return (
        round(cr * alpha + r * (1 - alpha)),
        round(cg * alpha + g * (1 - alpha)),
        round(cb * alpha + b * (1 - alpha)),
        255,
    )


def set_px(pixels, x: int, y: int, color: tuple[int, int, int, int]) -> None:
    if 0 <= y < len(pixels) and 0 <= x < len(pixels[0]):
        pixels[y][x] = blend(pixels[y][x], color)


def draw_line(pixels, x0: float, y0: float, x1: float, y1: float, color, width: int = 1) -> None:
    x0i, y0i, x1i, y1i = map(lambda v: int(round(v)), (x0, y0, x1, y1))
    dx = abs(x1i - x0i)
    dy = -abs(y1i - y0i)
    sx = 1 if x0i < x1i else -1
    sy = 1 if y0i < y1i else -1
    err = dx + dy
    x, y = x0i, y0i
    while True:
        for oy in range(-width, width + 1):
            for ox in range(-width, width + 1):
                if ox * ox + oy * oy <= width * width:
                    set_px(pixels, x + ox, y + oy, color)
        if x == x1i and y == y1i:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def draw_circle(pixels, cx: float, cy: float, radius: int, color, width: int = 1) -> None:
    cx_i, cy_i = int(round(cx)), int(round(cy))
    for y in range(cy_i - radius - width, cy_i + radius + width + 1):
        for x in range(cx_i - radius - width, cx_i + radius + width + 1):
            d = math.hypot(x - cx_i, y - cy_i)
            if radius - width <= d <= radius + width:
                set_px(pixels, x, y, color)


def draw_text(pixels, x: int, y: int, text: str, color, scale: int = 1) -> None:
    cursor = x
    for char in text.lower():
        glyph = FONT_5X7.get(char, FONT_5X7[" "])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    for oy in range(scale):
                        for ox in range(scale):
                            set_px(pixels, cursor + gx * scale + ox, y + gy * scale + oy, color)
        cursor += 6 * scale


def matching_pairs(
    centers: list[tuple[float, float, float]],
    vec: tuple[float, float],
    tolerance: float = 13.0,
) -> list[tuple[float, float, float, float]]:
    vx, vy = vec
    out = []
    for x1, y1, _ in centers:
        for x2, y2, _ in centers:
            if math.hypot((x2 - x1) - vx, (y2 - y1) - vy) <= tolerance:
                out.append((x1, y1, x2, y2))
    return out


def annotate(
    pixels: list[list[tuple[int, int, int, int]]],
    centers: list[tuple[float, float, float]],
    hvec: tuple[float, float],
    dvec: tuple[float, float],
    metrics: dict[str, float],
) -> list[list[tuple[int, int, int, int]]]:
    out = [row[:] for row in pixels]
    height, width = len(out), len(out[0])

    yellow = (255, 220, 0, 230)
    cyan = (0, 230, 255, 230)
    red = (255, 40, 40, 235)
    green = (30, 255, 90, 240)
    magenta = (255, 60, 210, 240)
    black = (0, 0, 0, 190)
    white = (255, 255, 255, 245)

    for x, y, _ in centers:
        draw_circle(out, x, y, 7, red, width=1)

    for x0, y0, x1, y1 in matching_pairs(centers, hvec):
        draw_line(out, x0, y0, x1, y1, yellow, width=2)
    for x0, y0, x1, y1 in matching_pairs(centers, dvec):
        draw_line(out, x0, y0, x1, y1, cyan, width=2)

    # Rulers: one x-axis atom spacing on top, one y-axis atom-spacing equivalent on left.
    x_step = metrics["x_pixels_per_atom"]
    y_step = metrics["y_pixels_per_atom_equiv"]
    draw_line(out, 5, 12, width - 6, 12, green, width=1)
    tick = 5.0
    while tick < width - 5:
        draw_line(out, tick, 6, tick, 18, green, width=1)
        tick += x_step
    draw_line(out, 12, 5, 12, height - 6, magenta, width=1)
    tick = 5.0
    while tick < height - 5:
        draw_line(out, 6, tick, 18, tick, magenta, width=1)
        tick += y_step

    # Small dark strips improve text readability without covering atoms too much.
    for yy in range(20, 43):
        for xx in range(3, min(width, 170)):
            set_px(out, xx, yy, black)
    for yy in range(height - 25, height - 3):
        for xx in range(3, min(width, 220)):
            set_px(out, xx, yy, black)

    draw_text(out, 8, 24, f"a={ATOM_SPACING_NM:.3f}nm", yellow, scale=1)
    draw_text(out, 8, 34, f"diag a={ATOM_SPACING_NM:.3f}nm", cyan, scale=1)
    draw_text(out, 8, height - 21, f"x={metrics['x_length_nm']:.3f}nm", white, scale=1)
    draw_text(out, 106, height - 21, f"y={metrics['y_length_nm']:.3f}nm", white, scale=1)
    return out


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Estimate STM x/y scan length from atom spacings.")
    parser.add_argument("image", nargs="?", default=str(script_dir / "11-1.png"), help="input PNG image")
    parser.add_argument("-o", "--output", default=None, help="annotated output PNG")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve() if args.output else image_path.with_name(image_path.stem + "_annotated.png")

    width, height, pixels = read_png_rgba(image_path)
    gray = grayscale(pixels)
    centers = detect_atom_centers(gray)
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
