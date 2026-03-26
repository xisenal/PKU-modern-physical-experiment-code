import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Heiti TC']
rcParams['axes.unicode_minus'] = False
currents = np.arange(3, 16)
data = {
    '6.74:1, 651 Pa': [2.86, 2.73, 2.40, 1.96, 1.52, 1.12, 0.67, 0.32, 0.13, np.nan, np.nan, np.nan, np.nan],
    '6.74:1, 470 Pa': [3.16, 3.43, 3.52, 3.40, 3.12, 2.77, 2.54, 2.28, 1.86, 1.72, 1.34, 1.22, np.nan],
    '6.74:1, 350 Pa': [np.nan, np.nan, 2.71, 2.99, 3.04, 3.02, 2.94, 2.81, 2.83, 2.64, 2.51, 2.40, 2.18],
    '5.71:1, 502 Pa': [3.00, 3.19, 2.82, 2.47, 2.13, 1.44, 0.81, 0.29, 0.03, np.nan, np.nan, np.nan, np.nan],
    '5.71:1, 353 Pa': [np.nan, np.nan, 4.36, 4.55, 4.46, 4.26, 3.95, 3.86, 3.77, 3.21, 2.84, 2.38, np.nan],
    '5.71:1, 251 Pa': [np.nan, np.nan, np.nan, np.nan, 3.46, 3.81, 3.83, 3.76, 3.91, 3.72, 3.68, 3.56, 3.50],
}
plt.figure(figsize=(10, 6))
for label, powers in data.items():
    powers = np.array(powers)
    mask = ~np.isnan(powers)
    plt.plot(currents[mask], powers[mask], marker='o', label=label)
    if np.any(mask):
        peak_index = np.nanargmax(powers)
        peak_current = currents[peak_index]
        peak_power = powers[peak_index]
        plt.scatter(peak_current, peak_power, color='red', s=50, zorder=5)
        plt.text(peak_current, peak_power + 0.1, f'峰值\n{peak_power:.2f}mW',
                 fontsize=8, ha='center', color='red')
plt.xlabel('工作电流 I (mA)')
plt.ylabel('激光功率 P (mW)')
plt.title('不同总压强与配气比下的激光功率-工作电流曲线')
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()