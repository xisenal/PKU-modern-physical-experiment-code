import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analy import read_dat_spectrum


def read_x_values(x_dat_path: Path) -> np.ndarray:
    x = np.loadtxt(x_dat_path, dtype=float)
    return np.asarray(x, dtype=float).reshape(-1)


def bin_spectrum(e: np.ndarray, counts: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(e, bins=bins, weights=counts)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, hist


def main():
    base_dir = Path(__file__).parent

    x_path = base_dir / "co60-x-calibrated.dat"
    if len(sys.argv) >= 2 and sys.argv[1].strip():
        x_path = Path(sys.argv[1].strip())

    e = read_x_values(x_path)

    bins = np.linspace(0.0, 1.5, 110)
    y_paths = [
        base_dir / "co60.dat",
        base_dir / "cs137near.dat",
        base_dir / "cs137far.dat",
        base_dir / "co+cs.dat",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True, sharex=True, sharey=False)
    axes = axes.ravel()

    for ax, y_path in zip(axes, y_paths):
        counts = np.asarray(read_dat_spectrum(y_path), dtype=float)
        n = min(e.size, counts.size)
        centers, hist = bin_spectrum(e[:n], counts[:n], bins=bins)
        ax.step(centers, hist, where="mid", linewidth=1.0)
        ax.set_title(y_path.stem)
        ax.set_xlabel("E / MeV")
        ax.set_ylabel("entries")

    output_path = base_dir / "spectra-bin-2x2.png"
    fig.savefig(output_path, dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
