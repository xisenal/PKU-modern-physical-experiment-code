import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq

x=[6.6,6.7,6.8,6.9,7.0,7.1,7.2]
y=[6383,8417,10168,10570,9691,7404,4574]

x=np.asarray(x,dtype=float)
y=np.asarray(y,dtype=float)
idx=np.argsort(x)
x=x[idx]
y=y[idx]
s=UnivariateSpline(x,y,s=0,k=3)
xs=np.linspace(x.min(),x.max(),400)
ys=s(xs)
d1=s.derivative()
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

candidates_x = np.concatenate([np.asarray([x.min(), x.max()]), roots])
candidates_y = s(candidates_x)
x_peak = float(candidates_x[int(np.argmax(candidates_y))])
y_peak=float(s(x_peak))
plt.figure(figsize=(8,5))
plt.scatter(x,y,color="k",s=40,label="data")
plt.plot(xs,ys,color="C1",label="spline")
plt.scatter([x_peak],[y_peak],color="r",zorder=5,label=f"peak x={x_peak:.3f}, y={y_peak:.1f}")
plt.axvline(x_peak,color="r",linestyle="--",alpha=0.3)
plt.xlabel("U_th/V")
plt.ylabel("Entries")
plt.title("Spline fit and peak")
# plt.grid(True,alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("s1-1-spline.png",dpi=200)
plt.show()
