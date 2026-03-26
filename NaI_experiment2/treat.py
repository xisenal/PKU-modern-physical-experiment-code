import pandas as pd
import matplotlib.pyplot as plt

def main():
    excel_path = "data.xlsx"
    df = pd.read_excel(excel_path, sheet_name=0)

    if df.shape[1] < 2:
        raise ValueError("data.xlsx 至少需要两列数据")

    x = df.iloc[:, 0]
    y = df.iloc[:, 1]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="·", linewidth=1)
    plt.xlabel("U_th/V")
    plt.ylabel("entries")
    plt.title(f"U_th vs entries")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig("plot.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()