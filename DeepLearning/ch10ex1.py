# 畳み込み演算の例：平滑化（ノイズを取り除く）
import numpy as np
import matplotlib.pyplot as plt

kk = 1000
time = np.arange(kk)

# 入力データを用意
x = np.sin(time/100.0*np.pi) + (np.random.rand(kk)*0.2)

# インパルス応答を用意
h = np.array([1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10])
nn = len(h)

# 畳み込み演算により出力データを計算
xx = np.zeros([kk, nn])
for n in range(nn):
    xx[n:kk,n] = x[0:kk-n]
y = np.dot(h, xx.T)

# グラフを作成
plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(time, x)
plt.title('Input')
plt.subplot(212)
plt.plot(time, y)
plt.title('Output')
plt.subplots_adjust(hspace=0.5)
plt.savefig("ch10ex1fig1.png")