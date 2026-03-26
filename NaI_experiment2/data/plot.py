import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analy import read_dat_spectrum


def read_x_values(x_dat_path: Path) -> np.ndarray:
    x = np.loadtxt(x_dat_path, dtype=float)
    x = np.asarray(x, dtype=float).reshape(-1)
    return x


def main():
    base_dir = Path(__file__).parent
    x_path = base_dir / "co60-x-calibrated.dat"

    if len(sys.argv) >= 2 and sys.argv[1].strip():
        x_path = Path(sys.argv[1].strip())

    x = read_x_values(x_path)

    y_paths = [
        base_dir / "co60.dat",
        base_dir / "cs137near.dat",
        base_dir / "cs137far.dat",
        base_dir / "co+cs.dat",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes = axes.ravel()

    for ax, y_path in zip(axes, y_paths):
        y = np.asarray(read_dat_spectrum(y_path), dtype=float)
        n = min(x.size, y.size)
        ax.plot(x[:n], y[:n], linewidth=0.8)
        ax.set_title(y_path.stem)
        ax.set_xlabel("E / MeV")
        ax.set_ylabel("entries")


    output_path = base_dir / "spectra-2x2.png"
    fig.savefig(output_path, dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
