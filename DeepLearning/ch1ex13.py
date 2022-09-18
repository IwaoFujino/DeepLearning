# Matplotlibの基本：1枚の図に複数のグラフを配置
import numpy as np
import matplotlib.pyplot as plt

# データを用意
kk = 1000
time = np.arange(kk)
# 正弦波
sine = np.sin(time/100.0 * np.pi)
# 方形波
square = np.empty(kk)
tnum = np.arange(10)
for t in tnum:
    square[t*100:(t+1)*100] = np.where(time[t*100:(t+1)*100]<t*100+100/2, 1, 0)
# グラフを作成
plt.figure(figsize=(10, 6))
plt.suptitle("Two Plots in One Figure", fontsize=20)
plt.subplot(211)
plt.plot(time, sine)
plt.title('sine wave', fontsize=16)
plt.xlabel('time')
plt.ylabel('sine')
plt.grid(color="red", linestyle="dotted")
plt.subplot(212)
plt.plot(time, square)
plt.title('square wave', fontsize=16)
plt.xlabel('time')
plt.ylabel('square')
plt.grid(color="red", linestyle="dotted")
plt.subplots_adjust(hspace=0.5)
plt.savefig("ch1ex13fig1.png")
