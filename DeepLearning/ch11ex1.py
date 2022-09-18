# 再帰型システムの動作確認
# 入力信号：なし
import numpy as np
import matplotlib.pyplot as plt

kk = 100
time = np.arange(kk)
x = np.zeros(kk)
x[0] = 1.0
a = 1.0
b = np.array([0.99, 0.95, 0.8, 0.3])
# 出力信号を求める．
y = np.zeros([len(b),kk])
for i in range(len(b)):
    y[i, 0] = 0.0
    for k in range(0,kk):
        y[i, k] = a*x[k] + b[i]*y[i, k-1]
# グラフを作成
plt.figure(figsize=(10, 6))
linestyle = ["solid","dashed", "dashdot", "dotted"]
for i in range(len(b)):
    plt.plot(time, y[i,:], linestyle=linestyle[i], label="b="+str(b[i]))
plt.title('Output of a Simple Recurrent System', fontsize=20)
plt.xlabel('k', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.legend()
plt.savefig("ch11ex1fig1.png")