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

    datasets = [
        (base_dir / "co60.dat", "Co-60", "C0"),
        (base_dir / "cs137near.dat", "Cs-137 near", "C1"),
        (base_dir / "cs137far.dat", "Cs-137 far", "C2"),
        (base_dir / "co+cs.dat", "Co + Cs", "C3"),
    ]

    bins = np.linspace(0.0, 1.5, 110)

    plt.figure(figsize=(9.5, 6.0))
    for y_path, label, color in datasets:
        counts = np.asarray(read_dat_spectrum(y_path), dtype=float)
        n = min(e.size, counts.size)
        centers, hist = bin_spectrum(e[:n], counts[:n], bins=bins)
        plt.step(centers, hist, where="mid", linewidth=1.0, label=label, color=color)

    plt.xlabel("E / MeV")
    plt.ylabel("entries")
    plt.title("Binned spectra")
    plt.xlim(0.0, 1.5)
    plt.legend()
    plt.tight_layout()

    output_path = base_dir / "spectra-bin-all.png"
    plt.savefig(output_path, dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
