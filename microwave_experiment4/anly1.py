from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def configure_matplotlib_fonts() -> None:
    # 为 macOS 优先设置常见中文字体，避免中文标签显示为方框。
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

    file_path = Path(__file__).with_name("data.xlsx")

    # 第一页无表头，直接读取前四列并命名为 A/B/C/D。
    df = pd.read_excel(
        file_path,
        sheet_name=0,
        header=None,
        usecols="A:D",
        names=["A", "B", "C", "D"],
    )

    if df["A"].dropna().empty:
        raise ValueError("A 列为空，无法作为 x 轴绘图。")

    valid_b = df[["A", "B"]].dropna()
    valid_c = df[["A", "C"]].dropna()
    valid_d = df[["A", "D"]].dropna()

    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(right=0.82)

    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.15))
    ax3.spines["right"].set_visible(True)

    lines = []
    labels = []

    if not valid_b.empty:
        line_b, = ax1.plot(
            valid_b["A"],
            valid_b["B"],
            color="tab:blue",
            marker="o",
            linewidth=2.0,
            markersize=5,
            label="工作电流",
        )
        lines.append(line_b)
        labels.append(line_b.get_label())
        ax1.set_ylabel("工作电流 / mA", color="tab:blue", fontsize=12)
        ax1.tick_params(axis="y", labelcolor="tab:blue")

    if not valid_c.empty:
        line_c, = ax2.plot(
            valid_c["A"],
            valid_c["C"],
            color="tab:red",
            marker="^",
            linewidth=2.0,
            markersize=5,
            label="频率",
        )
        lines.append(line_c)
        labels.append(line_c.get_label())
        ax2.set_ylabel("频率 / GHz", color="tab:red", fontsize=12)
        ax2.tick_params(axis="y", labelcolor="tab:red")

    if not valid_d.empty:
        line_d, = ax3.plot(
            valid_d["A"],
            valid_d["D"],
            color="tab:green",
            marker="s",
            linewidth=2.0,
            markersize=5,
            label="功率",
        )
        lines.append(line_d)
        labels.append(line_d.get_label())
        ax3.set_ylabel("功率 / 格", color="tab:green", fontsize=12)
        ax3.tick_params(axis="y", labelcolor="tab:green")

    ax1.set_xlabel("电压 / V", fontsize=12)
    ax1.set_title("电压-电流-频率-功率关系图", fontsize=16, pad=14)
    ax1.grid(True, linestyle="--", alpha=0.35)

    if lines:
        ax1.legend(lines, labels, loc="best")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
