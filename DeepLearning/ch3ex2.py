# 確率勾配降下法による重み推定
import numpy as np

# データを取得する
seisekidata = np.load("sougouseiseki.npz")
kk = 100
xx = seisekidata["x"][0:kk]
yy = seisekidata["y"][0:kk]
xx1 = np.array(xx[:,0]).T
xx2 = np.array(xx[:,1]).T
yy = np.array(yy).T
print("xx1 shape =", xx1.shape)
print("xx2 shape =", xx2.shape)
print("yy shape =", yy.shape)
# 確率勾配降下法により，重みを計算
wwold1 = 0.0
wwold2 = 0.0
eta = 0.00017
for k, datak in enumerate(zip(xx1, xx2, yy)):
    x1 = datak[0]
    x2 = datak[1]
    y = datak[2]
    yesti = wwold1*x1+wwold2*x2
    error = y-yesti
    ww1 = wwold1+eta*error*x1
    ww2 = wwold2+eta*error*x2
    print("k={0:2d}  x1={1:3.0f}  x2={2:3.0f}  error={3:10.6f}  ww1={4:10.6f}  ww2={5:10.6f}".format(k, x1, x2, error, ww1, ww2))
    wwold1 = ww1
    wwold2 = ww2
