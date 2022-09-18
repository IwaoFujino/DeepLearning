# 2次元畳み込み演算の例：動作確認
# ベクトルに変換してからnp.dotを使う計算法
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
# カーネル（ソーベルフィルタ）
kx = np.array([[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
            ])
# 入力画像を読み込む．
imgin = misc.face(gray=True).astype(np.float32)
# 2次元畳み込み演算により出力画像を計算
h = kx.flatten()
nn = h.shape[0]
# 2次元畳み込み演算により出力画像を計算
kk1, kk2 = imgin.shape
kk = kk1*kk2
mm1, mm2 = kx.shape
# mm1，mm2は奇数
m1max = int((mm1-1)/2)
m2max = int((mm2-1)/2)
xx = np.zeros([kk1+mm1, kk2+mm2, mm1, mm2])
for m1 in range(mm1):
    for m2 in range(mm2):
        xx[m1:kk1+m1, m2:kk2+m2, m1, m2] = imgin[0:kk1, 0:kk2]
xxx = np.zeros([kk,nn])
for k1 in range(kk1):
    for k2 in range(kk2):
        xxflip = np.flip(xx[k1+m1max, k2+m2max, :, :])
        k = k1*kk2 + k2
        xxx[k] = xxflip.flatten()
y = np.dot(h, xxx.T)
y = np.where(y<0, 0, y)
# ベクトルから画像に戻す．
imgout = y.reshape([kk1, kk2])
# 画像を表示
plt.figure(figsize=(14, 6))
plt.subplot(121)
plt.gray()
plt.imshow(imgin)
plt.title("Original Image")
plt.subplot(122)
plt.gray()
plt.imshow(imgout)
plt.title("Processed Image")
plt.savefig("ch12ex2fig1.png")