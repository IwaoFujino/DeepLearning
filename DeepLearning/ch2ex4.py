# 勾配降下法により，2の平方根を求める + グラフ作成
import matplotlib.pyplot as plt
import numpy as np

kk = 10
x = np.empty(kk+1)
x[0] = 3
eta = 0.01
for k in range(kk):
    delta = -4.0*x[k]*(2 - x[k]**2)
    x[k+1] = x[k] - eta*delta
# 損失関数 y = (2 - x^2)^2 の値を計算
y = (2 - x**2) ** 2
# xとyの値を表示
for k, data in  enumerate(zip(x,y)):
    print("k={0:4d}  x={1:15.12f}  Loss={2:15.12f}".format(k, data[0], data[1]))
# グラフを作成
plt.figure(figsize=(10, 6))
plt.scatter(x, y, c="red")
for no, data in enumerate(zip(x,y)):
    plt.annotate(str(no), data)
# 独立変数xの値の配列をつくる.
xx = np.arange(0, 3.1, 0.1)
# 損失関数 yy = (2 - xx^2) ^2 の値を計算
yy = (2 - xx**2) ** 2
plt.plot(xx, yy)
plt.title("Loss During Gradient Descending", fontsize=20)
plt.xlabel("x", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.savefig("ch2ex4fig1.png")