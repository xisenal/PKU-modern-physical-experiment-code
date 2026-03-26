import numpy as np
import matplotlib.pyplot as plt

x=[6.698630842823,5.860613875323,1.040273126277,3.404229570784]
y=[1.33,1.17,0.184,0.662]

x=np.asarray(x,dtype=float)
y=np.asarray(y,dtype=float)

a,b=np.polyfit(x,y,1)

r=float(np.corrcoef(x,y)[0,1])

xx=np.linspace(x.min(),x.max(),200)
yy=a*xx+b

plt.figure(figsize=(7,5))
plt.scatter(x,y,color="k",s=30,label="data")
plt.plot(xx,yy,color="C1",label=f"fit: y={a:.4g}x{b:.4g}; r={r:.6g}")
plt.xlabel("U/V")
plt.ylabel("E/MeV")
plt.title("Linear fit U vs E")
plt.legend()
# plt.grid(True,alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("s2-2-linear.png",dpi=200)
plt.show()

print(f"a={a}")
print(f"b={b}")
print(f"r={r}")
