# ミニバッチ勾配降下法による重み推定 ＋ グラフ作成
import numpy as np
import matplotlib.pyplot as plt

# データを取得
seisekidata = np.load("sougouseiseki.npz")
xx = seisekidata["x"]
kk, mm = xx.shape
print("kk=", kk)
print("mm=", mm)
kk = 1000
xx = seisekidata["x"][0:kk]
yy = seisekidata["y"][0:kk]
xx1 = np.array(xx[:, 0]).reshape((kk, 1))
xx2 = np.array(xx[:, 1]).reshape((kk, 1))
yy = np.array(yy).reshape((kk, 1))
print("xx1 shape =", xx1.shape)
print("xx2 shape =", xx2.shape)
print("yy shape =", yy.shape)
# 確率勾配降下法により，重みを計算
eta = 0.00002
tt = 10
bb = int(kk/tt)
error = np.empty(bb)
sqerr = np.empty(bb)
ww1 = np.empty(bb+1)
ww2 = np.empty(bb+1)
ww1[0] = 0.0
ww2[0] = 0.0
errorb = np.empty(tt)
for b in range(bb):
	xx1b = xx1[b*tt:(b+1)*tt]
	xx2b = xx2[b*tt:(b+1)*tt]
	yyb = yy[b*tt:(b+1)*tt]
	delta1 = 0.0
	delta2 = 0.0
	for t in range(tt):
		k = b*tt+t
		x1 = xx1b[t]
		x2 = xx2b[t]
		y = yyb[t]
		yesti = ww1[b]*x1 + ww2[b]*x2
		errorb[t] = y - yesti
		delta1 = delta1 + errorb[t]*xx1b[t]
		delta2 = delta2 + errorb[t]*xx2b[t]
	error[b] = np.mean(errorb)
	sqerr[b] = error[b]*error[b]
	ww1[b+1] = ww1[b] + eta*delta1
	ww2[b+1] = ww2[b] + eta*delta2
	print("b={0:2d} error={1:10.6f}  ww1={2:10.6f}  ww2={3:10.6f}".format(b, error[b], ww1[b], ww2[b]))
# グラフを作成
x = np.arange(0, bb, 1)
plt.figure(figsize=(10, 6))
plt.plot(x, sqerr)
plt.title("Value of Squared Error", fontsize=20)
plt.xlabel("Batch", fontsize=16)
plt.ylabel("Squared Error", fontsize=16)
plt.savefig("ch3ex4fig1.png")
x = np.arange(0, bb+1, 1)
plt.figure(figsize=(10, 6))
plt.plot(x, ww1, color="red", linestyle="-", label="ww1")
plt.plot(x, ww2, color="blue", linestyle="--", label="ww2")
plt.title("Value of Weights", fontsize=20)
plt.xlabel("Batch", fontsize=16)
plt.ylabel("Weight", fontsize=16)
plt.legend()
plt.savefig("ch3ex4fig2.png")