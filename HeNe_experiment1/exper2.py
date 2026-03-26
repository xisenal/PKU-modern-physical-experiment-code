import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Heiti TC']
rcParams['axes.unicode_minus'] = False
currents = np.arange(3, 16)
data = {
    '6.74:1': {
        651: [2.86, 2.73, 2.40, 1.96, 1.52, 1.12, 0.67, 0.32, 0.13, np.nan, np.nan, np.nan, np.nan],
        470: [3.16, 3.43, 3.52, 3.40, 3.12, 2.77, 2.54, 2.28, 1.86, 1.72, 1.34, 1.22, np.nan],
        350: [np.nan, np.nan, 2.71, 2.99, 3.04, 3.02, 2.94, 2.81, 2.83, 2.64, 2.51, 2.40, 2.18],
        286: [np.nan, np.nan, np.nan, 2.04, 2.20, 2.26, 2.23, 2.30, 2.22, 2.16, 2.18, 2.10, 2.03],
        203: [np.nan, np.nan, np.nan, np.nan, 0.18, 0.31, 0.35, 0.44, 0.48, 0.52, 0.57, 0.51, 0.46],
    },
    '5.71:1': {
        502: [3.00, 3.19, 2.82, 2.47, 2.13, 1.44, 0.81, 0.29, 0.03, np.nan, np.nan, np.nan, np.nan],
        353: [np.nan, np.nan, 4.36, 4.55, 4.46, 4.26, 3.95, 3.86, 3.77, 3.21, 2.84, 2.38, np.nan],
        251: [np.nan, np.nan, np.nan, np.nan, 3.46, 3.81, 3.83, 3.76, 3.91, 3.72, 3.68, 3.56, 3.50],
    }
}
peak_currents = {}
for ratio, pressures in data.items():
    peak_currents[ratio] = []
    sorted_pressures = sorted(pressures.keys(), reverse=True)
    for p in sorted_pressures:
        powers = np.array(pressures[p])
        mask = ~np.isnan(powers)
        if np.any(mask):
            peak_index = np.nanargmax(powers)
            peak_currents[ratio].append((p, currents[peak_index]))
plt.figure(figsize=(8,5))
for ratio, values in peak_currents.items():
    pressures, peak_I = zip(*values)
    plt.plot(pressures, peak_I, marker='o', label=f'配气比 {ratio}')
plt.xlabel('总压强 P (Pa)')
plt.ylabel('峰值功率对应的工作电流 I_peak (mA)')
plt.title('峰值功率对应的工作电流随总压强变化')
plt.gca().invert_xaxis()  # 压强从高到低显示
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()