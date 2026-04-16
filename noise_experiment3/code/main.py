from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from read_spectrum import read_spectrum


NOISE_DIR = Path(__file__).resolve().parents[1]
LAB1_DIR = NOISE_DIR / "20260415" / "lab1"
FREQ_MIN = 3000
FREQ_MAX = 9000
RESISTANCE_MIN = 1e3
RESISTANCE_MAX = 99e3
ROBUST_ITERATIONS = 6
ROBUST_C = 2.5


def read_resistance(path):
    with Path(path).open("r", encoding="utf-8") as file:
        lines = file.readlines()

    if len(lines) < 3:
        raise ValueError(f"Cannot read resistance from the third line of {path}")

    line = lines[2].strip()
    match = re.search(r"R\s*=\s*([0-9.]+)\s*([kKmM]?)\s*Ohm", line)
    if not match:
        raise ValueError(f"Cannot parse resistance from line: {line}")

    value = float(match.group(1))
    prefix = match.group(2).lower()
    if prefix == "k":
        value *= 1e3
    elif prefix == "m":
        value *= 1e6

    return value


def robust_mean(values):
    mean = np.mean(values)

    for _ in range(ROBUST_ITERATIONS):
        residual = values - mean
        mad = np.median(np.abs(residual))
        scale = 1.4826 * mad
        if scale == 0:
            break

        normalized_residual = residual / (ROBUST_C * scale)
        weights = 1 / (1 + normalized_residual**2)
        mean = np.sum(weights * values) / np.sum(weights)

    return mean


def noise_level_from_spectrum(path):
    spectrum = read_spectrum(path)
    freq = spectrum["freq"]
    s_xy = spectrum["S_xy"]

    positive_freq = freq > 0
    freq = freq[positive_freq]
    s_xy = s_xy[positive_freq]

    selected = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
    if not np.any(selected):
        raise ValueError(f"No frequency points found between {FREQ_MIN} Hz and {FREQ_MAX} Hz in {path}")

    return robust_mean(s_xy[selected])


def main():
    spectrum_paths = sorted(LAB1_DIR.glob("*spectrum.dat"))
    if not spectrum_paths:
        raise FileNotFoundError(f"No spectrum files found in {LAB1_DIR}")

    resistances = []
    noise_levels = []

    for path in spectrum_paths:
        resistance = read_resistance(path)
        if resistance < RESISTANCE_MIN or resistance > RESISTANCE_MAX:
            continue

        noise_level = noise_level_from_spectrum(path)
        resistances.append(resistance)
        noise_levels.append(noise_level)
        print(f"R = {resistance:g} Ohm, mean S_xy = {noise_level:.6e}")

    resistances = np.array(resistances)
    noise_levels = np.array(noise_levels)
    order = np.argsort(resistances)
    resistances = resistances[order]
    noise_levels = noise_levels[order]
    noise_levels_squared = noise_levels**2

    log_resistances = np.log(resistances)
    log_noise_levels_squared = np.log(noise_levels_squared)

    result = linregress(log_resistances, log_noise_levels_squared)
    a = result.slope
    b = result.intercept
    r = result.rvalue
    log_resistance_fit = np.linspace(min(log_resistances), max(log_resistances), 200)
    log_noise_fit = a * log_resistance_fit + b

    print("resistances =", resistances)
    print("noise_levels =", noise_levels)
    print("noise_levels_squared =", noise_levels_squared)
    print(f"linear fit: log(S_xy^2) = {a:.6e} * log(R) + {b:.6e}")
    print(f"r = {r:.6f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(log_resistances, log_noise_levels_squared, label="Data")
    plt.plot(log_resistance_fit, log_noise_fit, label="Linear Fit")
    plt.xlabel("log(Resistance (Ohm))")
    plt.ylabel(f"log(Mean $S_{{xy}}^2$ from {FREQ_MIN}-{FREQ_MAX} Hz)")
    plt.title("Lab1 Log-Log Noise Level Fit")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
