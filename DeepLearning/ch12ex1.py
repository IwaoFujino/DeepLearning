# 2次元畳み込み演算の例：入力行列の対象範囲の確認
import numpy as np

# カーネル
kx = np.array([[11, 12, 13],
            [21, 22, 23],
            [31, 32, 33],
            ])
# 入力画像
imgin = np.array([[11, 12, 13, 14],
            [21, 22, 23, 24],
            [31, 32, 33, 34],
            [41, 42, 43, 44],
            [51, 52, 53, 54],               
            ])
h = kx.flatten()
nn = h.shape[0]
# 2次元畳み込み演算により出力画像を計算
kk1, kk2 = imgin.shape
kk = kk1*kk2
mm1, mm2 = kx.shape
# mm1，mm2は奇数
m1max = int((mm1-1)/2)
m2max = int((mm2-1)/2)
xx = np.zeros([kk1+mm1,kk2+mm2,mm1,mm2])
for m1 in range(mm1):
    for m2 in range(mm2):
        xx[m1:kk1+m1,m2:kk2+m2,m1,m2]=imgin[0:kk1, 0:kk2]
        print("m1=",m1, "m2=",m2, "................")
        print(xx[:,:,m1,m2])
xxx = np.zeros([kk,nn])
for k1 in range(kk1):
    for k2 in range(kk2):
        imgwin = np.flip(xx[k1+m1max,k2+m2max,:,:])
        print("k1=",k1, "k2=", k2, "imgin >>>")
        print(imgwin)
        k = k1*kk2 + k2
        xxx[k] = imgwin.flatten()
y = np.dot(h, xxx.T)
y = np.where(y<0, 0, y)
# ベクトルから画像に戻す.
imgout = y.reshape([kk1,kk2])
# 処理結果を表示
print("カーネル配列：")
print(kx)
print("処理前の配列：")
print(imgin)
print("処理後の配列：")
print(imgout)
print("処理完了")