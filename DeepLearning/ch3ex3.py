# 確率勾配降下法による重み推定＋グラフ作成
import numpy as np
import matplotlib.pyplot as plt

# データを取得
seisekidata = np.load("sougouseiseki.npz")
kk = 100
xx = seisekidata["x"][0:kk]
yy = seisekidata["y"][0:kk]
xx1 = np.array(xx[:, 0]).reshape((kk, 1))
xx2 = np.array(xx[:, 1]).reshape((kk, 1))
yy = np.array(yy).reshape((kk, 1))
print("xx1 shape =", xx1.shape)
print("xx2 shape =", xx2.shape)
print("yy shape =", yy.shape)
# 確率勾配降下法により，重みを計算
eta = 0.0002
error = np.empty(len(yy))
sqerr = np.empty(len(yy))
ww1 = np.empty(len(yy)+1)
ww2 = np.empty(len(yy)+1)
ww1[0] = 0.0
ww2[0] = 0.0
for k in range(kk):
    x1 = float(xx1[k])
    x2 = float(xx2[k])
    y = float(yy[k])
    yesti = ww1[k]*x1 + ww2[k]*x2
    error[k] = y - yesti
    sqerr[k] = error[k]*error[k]
    ww1[k+1] = ww1[k] + eta*error[k]*x1
    ww2[k+1] = ww2[k] + eta*error[k]*x2
    print("k={0:2d}  x1={1:3.0f}  x2={2:3.0f}  error={3:10.6f}  ww1={4:10.6f}  ww2={5:10.6f}".format(k, x1, x2, error[k], ww1[k], ww2[k]))
# グラフを作成
x = np.arange(0, len(yy), 1)
plt.figure(figsize=(10, 6))
plt.plot(x, sqerr)
plt.title("Value of Squared Error", fontsize=20)
plt.xlabel("k", fontsize=16)
plt.ylabel("Squared error", fontsize=16)
plt.savefig("ch3ex3fig1.png")
plt.figure(figsize=(10, 6))
x = np.arange(0, len(yy)+1, 1)
plt.plot(x, ww1, color="red", linestyle="-", label="ww1")
plt.plot(x, ww2, color="blue", linestyle="--", label="ww2")
plt.title("Value of Weights", fontsize=20)
plt.xlabel("k", fontsize=16)
plt.ylabel("Weight", fontsize=16)
plt.legend()
plt.savefig("ch3ex3fig2.png")