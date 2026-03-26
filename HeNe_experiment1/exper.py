import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Heiti TC']
rcParams['axes.unicode_minus'] = False
data = {
    '配气比6.74:1, 总压强651 Pa': [2.86, 2.73, 2.40, 1.96, 1.52, 1.12, 0.67, 0.32, 0.13, np.nan, np.nan, np.nan, np.nan],
    '配气比6.74:1, 总压强470 Pa': [3.16, 3.43, 3.52, 3.40, 3.12, 2.77, 2.54, 2.28, 1.86, 1.72, 1.34, 1.22, np.nan],
    '配气比6.74:1, 总压强350 Pa': [np.nan, np.nan, 2.71, 2.99, 3.04, 3.02, 2.94, 2.81, 2.83, 2.64, 2.51, 2.40, 2.18],
    '配气比6.74:1, 总压强286 Pa': [np.nan, np.nan, np.nan, 2.04, 2.20, 2.26, 2.23, 2.30, 2.22, 2.16, 2.18, 2.10, 2.03],
    '配气比6.74:1, 总压强203 Pa': [np.nan, np.nan, np.nan, np.nan, 0.18, 0.31, 0.35, 0.44, 0.48, 0.52, 0.57, 0.51, 0.46],
    '配气比5.71:1, 总压强502 Pa': [3.00, 3.19, 2.82, 2.47, 2.13, 1.44, 0.81, 0.29, 0.03, np.nan, np.nan, np.nan, np.nan],
    '配气比5.71:1, 总压强353 Pa': [np.nan, np.nan, 4.36, 4.55, 4.46, 4.26, 3.95, 3.86, 3.77, 3.21, 2.84, 2.38, np.nan],
    '配气比5.71:1, 总压强251 Pa': [np.nan, np.nan, np.nan, np.nan, 3.46, 3.81, 3.83, 3.76, 3.91, 3.72, 3.68, 3.56, 3.50],
}
currents = np.arange(3, 16)
plt.figure(figsize=(10,6))
for label, powers in data.items():
    powers = np.array(powers)
    mask = ~np.isnan(powers)
    plt.plot(currents[mask], powers[mask], marker='o', label=label)
plt.xlabel('工作电流 I (mA)')
plt.ylabel('激光功率 (mW)')
plt.title('不同总压强与配气比下的激光功率-工作电流曲线')
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()