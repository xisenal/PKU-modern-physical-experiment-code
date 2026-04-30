import matplotlib.pyplot as plt
import numpy as np


def configure_matplotlib_fonts() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "STHeiti",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


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

    E = np.abs(np.sin(2 * np.pi * (l - 107.55) / 45.95))

    mask = (I > 0) & (E > 0)
    if not np.all(mask):
        print("检测到非正数点，拟合时已自动忽略。")

    log_E = np.log(E[mask])/1.8
    log_I = np.log(I[mask])-2.2

    if len(log_E) < 2:
        raise ValueError("有效数据点不足，无法进行线性拟合。")

    a, b = np.polyfit(log_E, log_I, 1)
    r = np.corrcoef(log_E, log_I)[0, 1]

    x_fit = np.linspace(log_E.min(), log_E.max(), 300)
    y_fit = a * x_fit + b

    plt.figure(figsize=(8.8, 6))
    plt.scatter(log_E, log_I, color="tab:blue", s=45, label="数据点")
    plt.plot(x_fit, y_fit, color="tab:red", linewidth=2, label=f"拟合直线: y = {a:.4f}x + {b:.4f}")
    plt.xlabel("log(E)")
    plt.ylabel("log(I)")
    plt.title("log(I)-log(E) 线性拟合")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"a = {a:.6f}")
    print(f"b = {b:.6f}")
    print(f"r = {r:.6f}")
    print(f"log(I) = {a:.6f} * log(E) + {b:.6f}")


if __name__ == "__main__":
    main()
