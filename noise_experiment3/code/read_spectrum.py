from pathlib import Path
import argparse

import numpy as np


NOISE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SPECTRUM = NOISE_DIR / "20260415" / "lab1" / "100000o120919115132spectrum.dat"


def spectrum_path(name):
    path = Path(name)
    if path.suffix == "":
        path = path.with_name(path.name + "spectrum.dat")

    if path.exists() or path.is_absolute():
        return path

    noise_relative_path = NOISE_DIR / path
    if noise_relative_path.exists():
        return noise_relative_path

    return path


def read_spectrum(name):
    """Read an xxspectrum.dat file and return its spectrum columns."""
    path = spectrum_path(name)

    with path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    data_start = None
    for index, line in enumerate(lines):
        if line.strip().startswith("Tableau"):
            data_start = index + 1
            break

    if data_start is None:
        raise ValueError(f"Cannot find 'Tableau:' data marker in {path}")

    data = np.loadtxt(lines[data_start:])
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 6:
        raise ValueError(f"Expected at least 6 data columns in {path}, got {data.shape[1]}")

    return {
        "freq": data[:, 0],
        "S_xy": data[:, 1],
        "ch1": data[:, 2],
        "ch2": data[:, 3],
        "S_xy_imag": data[:, 4],
        "S_xy_real": data[:, 5],
        "raw": data,
    }


def main():
    parser = argparse.ArgumentParser(description="Plot S_xy against freq from an xxspectrum.dat file.")
    parser.add_argument(
        "name",
        nargs="?",
        default=DEFAULT_SPECTRUM,
        help="Path to xxspectrum.dat, relative path under noise/, or a name stem.",
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    spectrum = read_spectrum(args.name)

    positive_freq = spectrum["freq"] > 0
    freq = spectrum["freq"][positive_freq]
    plots = [
        ("S_xy", "$S_{xy}$ psd (V/$\\sqrt{Hz}$)"),
        ("ch1", "ch1 (V/$\\sqrt{Hz}$)"),
        ("ch2", "ch2 (V/$\\sqrt{Hz}$)"),
        ("S_xy_imag", "$S_{xy}$ imaginary part (V$^2$/Hz)"),
        ("S_xy_real", "$S_{xy}$ real part (V$^2$/Hz)"),
    ]

    fig, axes = plt.subplots(len(plots), 1, figsize=(11, 14), sharex=True)
    for ax, (key, ylabel) in zip(axes, plots):
        ax.plot(freq, spectrum[key][positive_freq])
        ax.set_xscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)

    axes[-1].set_xlabel("freq (Hz)")
    fig.suptitle("Spectrum vs freq", y=0.995)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
