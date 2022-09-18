# 2変数関数の勾配降下法 + グラフ作成
import numpy as np
import matplotlib.pyplot as plt

# 勾配降下法（変数x，yの更新）
kk = 100
x = np.empty(kk+1)
y = np.empty(kk+1)
x[0] = 1
y[0] = 1
eta = 0.2
for k in range(kk):
    deltax = np.exp(-( 3*x[k]**2+y[k]**2)/5)*6*x[k]/5
    deltay = np.exp(-( 3*x[k]**2+y[k]**2)/5)*2*y[k]/5
    x[k+1] = x[k] - eta*deltax
    y[k+1] = y[k] - eta*deltay
# 関数zの値を計算
z = 1.0 - np.exp(-( 3*x**2+y**2)/5)
# x,y,zの値を表示
for k, data in  enumerate(zip(x,y,z)):
    print("k={0:4d}  x={1:15.12f}  y={2:15.12f}  z={3:15.12f}".format(k, data[0], data[1], data[2]))
# x, y, zのグラフを作成
plt.figure(figsize=(10, 6))
k = np.arange(0, kk+1)
plt.plot(k, x, linestyle = "dashdot", label = "x")
plt.plot(k, y, linestyle = "dashed", label = "y")
plt.plot(k, z, linestyle = "solid", label = "z")
plt.title("Value of x, y, z During Gradient Descending", fontsize=20)
plt.xlabel("k", fontsize=16)
plt.ylabel("Value", fontsize=16)
plt.legend()
plt.savefig("ch2ex5fig1.png")