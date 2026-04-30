import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.interpolate import make_interp_spline
except ImportError as exc:
    raise ImportError("运行此脚本需要 scipy，请在你的 .venv 中安装 scipy 后再运行。") from exc


def configure_matplotlib_fonts() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "STHeiti",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def find_crossings(x: np.ndarray, y: np.ndarray, level: float) -> list[float]:
    shifted = y - level
    crossings: list[float] = []

    for i in range(len(x) - 1):
        y0 = shifted[i]
        y1 = shifted[i + 1]

        if y0 == 0:
            crossings.append(float(x[i]))
            continue

        if y0 * y1 < 0:
            ratio = (level - y[i]) / (y[i + 1] - y[i])
            crossings.append(float(x[i] + ratio * (x[i + 1] - x[i])))

    return crossings


def main() -> None:
    configure_matplotlib_fonts()

    l = np.array(
        [108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131],
        dtype=float,
    )
    I = np.array(
        [3.9, 6.1, 11.1, 18.5, 28.5, 32.1, 49, 62.5, 72, 80.9, 86.5, 89.6, 89.1, 86, 80.2, 71.7, 61.8, 51.3, 40.6, 30.8, 21.8, 13.2, 8.1, 4.1],
        dtype=float,
    )

    if len(l) != len(I):
        raise ValueError("l 和 I 的长度不一致，无法一一对应绘图。")

    spline = make_interp_spline(l, I, k=3)
    l_dense = np.linspace(l.min(), l.max(), 2000)
    I_smooth = spline(l_dense)

    baseline = float(I.min())
    peak = float(I.max())
    half_max = baseline + (peak - baseline) / 2

    crossings = find_crossings(l_dense, I_smooth, half_max)
    if len(crossings) < 2:
        raise RuntimeError("未找到两个半高交点，无法计算半高全宽。")

    left_cross = crossings[0]
    right_cross = crossings[-1]
    fwhm = right_cross - left_cross

    plt.figure(figsize=(10, 6))
    plt.plot(l_dense, I_smooth, color="tab:blue", linewidth=2.2, label="样条拟合曲线")
    plt.scatter(l, I, color="tab:orange", s=40, zorder=3, label="原始数据点")
    plt.axhline(half_max, color="tab:red", linestyle="--", linewidth=1.8, label=f"半高 I = {half_max:.2f}")
    plt.vlines(
        [left_cross, right_cross],
        ymin=baseline,
        ymax=half_max,
        color="tab:green",
        linestyle=":",
        linewidth=1.6,
        label="半高交点",
    )
    plt.scatter([left_cross, right_cross], [half_max, half_max], color="tab:green", zorder=4)
    plt.annotate(f"左交点: l = {left_cross:.2f}", xy=(left_cross, half_max), xytext=(left_cross - 3.2, half_max + 8))
    plt.annotate(f"右交点: l = {right_cross:.2f}", xy=(right_cross, half_max), xytext=(right_cross - 1.0, half_max + 8))
    plt.annotate(
        f"FWHM = {fwhm:.2f}",
        xy=((left_cross + right_cross) / 2, half_max),
        xytext=((left_cross + right_cross) / 2 - 1.2, half_max - 18),
        arrowprops={"arrowstyle": "->", "color": "black"},
    )

    plt.xlabel("l/mm")
    plt.ylabel("I/格")
    plt.title("I 关于 l 的样条拟合与半高全宽")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"半高所在的 I 高度: {half_max:.4f}")
    print(f"左侧半高交点: l = {left_cross:.4f}")
    print(f"右侧半高交点: l = {right_cross:.4f}")
    print(f"半高全宽 FWHM: {fwhm:.4f}")


if __name__ == "__main__":
    main()
