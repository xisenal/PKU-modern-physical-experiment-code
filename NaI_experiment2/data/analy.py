import sys
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks, savgol_filter


def read_dat_spectrum(dat_path: Union[str, Path]) -> list[float]:
    dat_path = Path(dat_path)
    bs = dat_path.read_bytes()

    try:
        text = bs.decode("utf-8")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        y = [float(v) for v in lines]
        if y:
            return y
    except UnicodeDecodeError:
        pass

    if len(bs) < 2:
        raise ValueError(f"{dat_path} 文件太小，无法解析")

    n = int(np.frombuffer(bs, dtype="<u2", count=1)[0])
    need = 2 + n * 4
    if len(bs) < need:
        raise ValueError(f"{dat_path} 数据长度不足：header 指示 n={n}，但文件只有 {len(bs)} bytes")

    y = np.frombuffer(bs, dtype="<u4", offset=2, count=n).astype(float)
    return y.tolist()


def _parabolic_peak_x(x: np.ndarray, y: np.ndarray, i: int) -> float:
    if i <= 0 or i >= len(x) - 1:
        return float(x[i])
    x1, x2, x3 = float(x[i - 1]), float(x[i]), float(x[i + 1])
    y1, y2, y3 = float(y[i - 1]), float(y[i]), float(y[i + 1])
    denom = (y1 - 2 * y2 + y3)
    if denom == 0:
        return x2
    delta = 0.5 * (y1 - y3) / denom
    return x2 + delta * (x3 - x2)


def find_two_rightmost_peaks_channel(y: np.ndarray) -> Tuple[float, float]:
    n = y.size
    x = np.arange(1, n + 1, dtype=float)

    window = min(61, n if n % 2 == 1 else n - 1)
    window = max(window, 7)
    if window >= n:
        window = n - 1 if (n - 1) % 2 == 1 else n - 2
    if window < 7:
        raise ValueError("数据点过少，无法自动找峰")

    y_smooth = savgol_filter(y, window_length=window, polyorder=3, mode="interp")
    y_range = float(np.max(y_smooth) - np.min(y_smooth))
    distance = max(5, n // 80)

    peaks = np.asarray([], dtype=int)
    for ratio in (0.06, 0.04, 0.03, 0.02, 0.015, 0.01, 0.006, 0.004):
        prom = max(1.0, ratio * y_range)
        peaks, _ = find_peaks(y_smooth, prominence=prom, distance=distance)
        if peaks.size >= 2:
            break
    if peaks.size < 2:
        raise ValueError("自动找峰失败：未能找到两个峰，请检查数据或调整算法参数")

    peaks = peaks[np.argsort(peaks)]
    p_left, p_right = int(peaks[-2]), int(peaks[-1])
    ch_left = _parabolic_peak_x(x, y_smooth, p_left)
    ch_right = _parabolic_peak_x(x, y_smooth, p_right)
    return float(ch_left), float(ch_right)


def calibrate_energy_axis_from_two_peaks(x_channel: np.ndarray, ch_left: float, ch_right: float, e_left: float = 1.17, e_right: float = 1.33) -> Tuple[float, float, np.ndarray]:
    if ch_right == ch_left:
        raise ValueError("两个峰的通道位置相同，无法标定")
    a = (e_right - e_left) / (ch_right - ch_left)
    b = e_left - a * ch_left
    e = a * x_channel + b
    return float(a), float(b), e


def write_dat_values_text(values: np.ndarray, output_path: Union[str, Path]):
    output_path = Path(output_path)
    text = "\n".join(f"{float(v):.12g}" for v in values) + "\n"
    output_path.write_text(text, encoding="utf-8")


def plot_spectrum_spline(y: list[float], output_path: str = "co60-spline.png", show: bool = True):
    n = len(y)
    if n < 4:
        raise ValueError("数据点过少，无法进行三次样条拟合（至少 4 个点）")

    x = np.arange(1, n + 1, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    s = UnivariateSpline(x, y_arr, s=0, k=3)
    xs = np.linspace(x.min(), x.max(), 4000)
    ys = s(xs)

    plt.figure(figsize=(9, 5))
    plt.plot(x, y_arr, color="k", linewidth=0.8, alpha=0.6, label="data")
    plt.plot(xs, ys, color="C1", linewidth=1.2, label="spline")
    plt.xlabel("channel")
    plt.ylabel("entries")
    plt.title("Spectrum (spline fit)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    if show:
        plt.show()


def plot_calibrated_spectrum(y: list[float], output_path: str = "co60-calibrated.png", x_output_path: Union[str, Path] = "co60-x-calibrated.dat", show: bool = True):
    y_arr = np.asarray(y, dtype=float)
    n = y_arr.size
    x_channel = np.arange(1, n + 1, dtype=float)

    ch_left, ch_right = find_two_rightmost_peaks_channel(y_arr)
    a, b, e = calibrate_energy_axis_from_two_peaks(x_channel, ch_left, ch_right, e_left=1.17, e_right=1.33)
    write_dat_values_text(e, x_output_path)

    s = UnivariateSpline(e, y_arr, s=0, k=3)
    es = np.linspace(float(e.min()), float(e.max()), 4000)
    ys = s(es)

    y_left = float(s(1.17))
    y_right = float(s(1.33))

    plt.figure(figsize=(9, 5))
    # plt.scatter(e, y_arr, color="k", s=5, label="data")
    plt.plot(es, ys, color="C1", linewidth=1.2, label="spline")
    plt.scatter([1.17, 1.33], [y_left, y_right], color="r", s=60, marker="^", zorder=5, label="calibration peaks")
    plt.annotate(f"(1.17, ch={ch_left:.2f})", xy=(1.17, y_left), xytext=(6, 10), textcoords="offset points", fontsize=9, color="r")
    plt.annotate(f"(1.33, ch={ch_right:.2f})", xy=(1.33, y_right), xytext=(6, 10), textcoords="offset points", fontsize=9, color="r")
    plt.xlabel("E / MeV")
    plt.ylabel("entries")
    plt.title(f"Co60 calibrated spectrum")
    plt.ylim(-300,4800)
    plt.xlim(-0.1,1.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    if show:
        plt.show()

    print(f"ch_left={ch_left}, E_left=1.17")
    print(f"ch_right={ch_right}, E_right=1.33")
    print(f"a={a}")
    print(f"b={b}")


def main():
    dat_path = Path(__file__).with_name("co60.dat")
    if len(sys.argv) >= 2 and sys.argv[1].strip():
        dat_path = Path(sys.argv[1].strip())

    y = read_dat_spectrum(dat_path)
    plot_calibrated_spectrum(y, output_path="co60-calibrated.png", x_output_path=Path(__file__).with_name("co60-x-calibrated.dat"), show=True)


if __name__ == "__main__":
    main()
