import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq


def plot_spline_peak(x_list, y_list, xlabel="U_th/V", ylabel="entries", title="U_th vs entries (spline peak)", ylim=None, output_path=None, show=True):
    x = np.asarray(x_list, dtype=float)
    y = np.asarray(y_list, dtype=float)
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    s = UnivariateSpline(x, y, s=0, k=3)
    xs = np.linspace(x.min(), x.max(), 400)
    ys = s(xs)
    d1 = s.derivative()
    d2 = d1.derivative()
    x_dense = np.linspace(x.min(), x.max(), 2000)
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
    roots = roots[(x.min() <= roots) & (roots <= x.max())]
    if roots.size:
        roots = np.unique(np.round(roots, 12))
    maxima_x = roots[d2(roots) < 0] if roots.size else np.asarray([], dtype=float)
    if maxima_x.size:
        vals = s(maxima_x)
        x_peak = float(maxima_x[int(np.argmax(vals))])
    else:
        x_peak = float(xs[int(np.argmax(ys))])
    y_peak = float(s(x_peak))
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, color="k", s=10, label="data")
    plt.plot(xs, ys, color="C1", label="spline")
    plt.scatter([x_peak], [y_peak], color="r", s=60, marker="^", zorder=5, label="peak")
    plt.axvline(x_peak, color="r", linestyle="--", alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200)
    if show:
        plt.show()
    return x_peak, y_peak


if __name__ == "__main__":
    # Cs 全能峰
    x = [3.3, 3.35, 3.4, 3.45, 3.5]
    y = [16150, 20404, 22961, 21221, 17274]
    # Cs 全反射峰
    # x=[0.95,1,1.05,1.1,1.15]
    # y=[7450,7475,7564,7223,6735]
    # Co 第一能峰
    # x=[5.75,5.8,5.85,5.9,5.95]
    # y=[9291,9845,10370,10185,9694]
    # Co 第二能峰
    # x=[6.6,6.65,6.7,6.75,6.8]
    # y=[6720,7048,7341,6955,6280]
    xp, yp = plot_spline_peak(x, y, xlabel="U_th/V", ylabel="entries", title="U_th vs entries (spline peak)", output_path="s2-1-spline.png", show=True)
    print(xp, yp)
