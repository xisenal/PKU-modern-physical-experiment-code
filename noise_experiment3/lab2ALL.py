
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# 原始数据
# V = np.array([1.1835, 1.0656, 0.9019, 0.8304, 0.7368, 0.6201,
#               0.5375, 0.4121, 0.3241, 0.2142, 0.1489, 0.0496])
#
# std = np.array([1.52E-04, 1.48E-04, 1.42E-04, 1.36E-04, 1.27E-04,
#                 1.18E-04, 1.09E-04, 9.59E-05, 8.58E-05, 6.95E-05,
#                 5.77E-05, 3.26E-05])


V = np.array([0.9019, 0.8304, 0.7368, 0.6201,
              0.5375, 0.4121, 0.3241, 0.2142, 0.1489])

std = np.array([1.39E-04, 1.33E-04, 1.25E-04,
                1.16E-04, 1.09E-04, 9.59E-05, 8.58E-05, 6.95E-05,
                5.77E-05])



std=std**2

# 线性拟合
result = linregress(V, std)
a = result.slope
b = result.intercept
r = result.rvalue

print(f"a = {a}")
print(f"b = {b}")
print(f"r = {r}")

# 拟合直线
V_fit = np.linspace(min(V), max(V), 100)
std_fit = a * V_fit + b

# 作图
plt.figure(figsize=(8, 6))
fit_color = 'tab:blue'
plt.scatter(V, std, color=fit_color, label='Data')
plt.plot(V_fit, std_fit, color=fit_color, label='Linear Fit')
plt.xlabel('Voltage (V)')
plt.ylabel('Square of Standard Deviation')
plt.title('Linear Fit of Noise vs Voltage')
plt.legend()
# plt.grid()

plt.show()


print(f"e=a/100000={a/100000}")
