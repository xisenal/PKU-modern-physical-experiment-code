import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# 原始数据（R单位：Ω）

# R = np.array([100e3, 50e3, 20e3, 10e3, 1e3,
#               500, 200, 100, 50, 20, 10, 5])
#
# V = np.array([3.47E-05, 2.72E-05, 1.77E-05, 1.24E-05,
#               4.03E-06, 2.83E-06, 2.09E-06, 2.01E-06,
#               1.01E-06, 5.91E-07, 4.13E-07, 2.99E-07])

R = np.array([100e3, 50e3, 20e3, 10e3, 1e3,
              500, 200, 50])

V = np.array([4.24E-05, 2.92E-05, 1.87E-05, 1.34E-05,
              4.03E-06, 2.83E-06, 1.89E-06, 9.81E-07])


# 计算 V^2
V2 = V**2

log_R = np.log(R)
log_V2 = np.log(V2)

# 线性拟合 log(V^2) = a log(R) + b
result = linregress(log_R, log_V2)
a = result.slope
b = result.intercept
r = result.rvalue

print(f"斜率 a = {a:.3e}")
print(f"截距 b = {b:.3e}")
print(f"相关系数 r = {r:.6f}")

# 拟合曲线：在对数坐标中显示为直线
R_fit = np.logspace(np.log10(min(R)), np.log10(max(R)), 200)
V2_fit = np.exp(b) * R_fit**a

# 作图
plt.figure(figsize=(8, 6))
plt.scatter(R, V2, label='Data')
plt.plot(R_fit, V2_fit, label='Linear Fit')
plt.xscale('log')
plt.yscale('log')

plt.xlabel('Resistance (Ohm)')
plt.ylabel('$V^2$ (V$^2$)')
plt.title('Thermal Noise Fit: $V^2$ vs R')

plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.show()

print(f"k_B=e^b/(4*T)={10**-6*np.exp(b)/(4*300)}")
# print(f"k_B=a/(4T)={a/(4*300)}")
