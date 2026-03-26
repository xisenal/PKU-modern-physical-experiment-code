import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq


def find_local_maxima_x(spline, x_min: float, x_max: float, *, grid_n: int = 4000) -> np.ndarray:
    d1 = spline.derivative()
    d2 = d1.derivative()

    x_dense = np.linspace(x_min, x_max, grid_n)
    dy_dense = d1(x_dense)

    roots = []
    tol = 1e-12
    near_zero = np.where(np.abs(dy_dense) < tol)[0]
    roots.extend(x_dense[near_zero].tolist())

    sign = np.sign(dy_dense)
    sign[sign == 0] = np.nan
    for i in range(len(x_dense) - 1):
        if np.isnan(sign[i]) or np.isnan(sign[i + 1]):
            continue
        if sign[i] == sign[i + 1]:
            continue
        a = float(x_dense[i])
        b = float(x_dense[i + 1])
        try:
            roots.append(float(brentq(d1, a, b)))
        except ValueError:
            pass

    roots = np.asarray(roots, dtype=float)
    roots = roots[np.isfinite(roots)]
    roots = roots[(x_min <= roots) & (roots <= x_max)]
    if roots.size:
        roots = np.unique(np.round(roots, 12))

    if roots.size == 0:
        return np.asarray([], dtype=float)

    is_max = d2(roots) < 0
    return roots[is_max]


def main():
    excel_path = "data.xlsx"
    if len(sys.argv) >= 2 and sys.argv[1].strip():
        excel_path = sys.argv[1].strip()

    df = pd.read_excel(excel_path, sheet_name=0)
    if df.shape[1] < 2:
        raise ValueError(f"{excel_path} 至少需要两列数据")

    x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 4:
        raise ValueError("有效数据点过少，无法进行三次样条拟合（至少 4 个点）")

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    s = UnivariateSpline(x, y, s=0, k=3)
    xs = np.linspace(float(x.min()), float(x.max()), 600)
    ys = s(xs)

    maxima_x = find_local_maxima_x(s, float(x.min()), float(x.max()))
    maxima_y = s(maxima_x) if maxima_x.size else np.asarray([], dtype=float)

    if maxima_x.size:
        order = np.argsort(maxima_y)[::-1]
        maxima_x = maxima_x[order]
        maxima_y = maxima_y[order]

    peaks_x = maxima_x[:2]
    peaks_y = maxima_y[:2]

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="k", s=10, label="data")
    plt.plot(xs, ys, color="C1", label="spline")
    plt.ylim(-200,12000)

    if peaks_x.size:
        plt.scatter(peaks_x, peaks_y, color="r", s=60, marker="^", zorder=5, label="peaks")
        for i, (px, py) in enumerate(zip(peaks_x, peaks_y), start=1):
            plt.annotate(
                f"peak{i}\n({px:.4g}, {py:.4g})",
                xy=(float(px), float(py)),
                xytext=(6, 10),
                textcoords="offset points",
                fontsize=9,
                color="r",
            )

    plt.xlabel("U_th/V")
    plt.ylabel("entries")
    plt.title("U_th vs entries")
    # plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("treat-spline-two-peaks.png", dpi=200)
    plt.show()
    print(peaks_x, peaks_y)


if __name__ == "__main__":
    main()
