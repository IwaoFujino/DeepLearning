# Matplotlibの基本：折れ線グラフを作成
import numpy as np
import matplotlib.pyplot as plt

# データを用意
nn=8
x = np.arange(1, nn+1, 1)
y1 = np.array([20, 24, 25, 23, 25, 25, 22, 23])
y2 = np.array([10, 14, 15, 15, 15, 16, 13, 14])
# 折れ線グラフを作成
plt.figure(figsize=(10, 6))
plt.plot(x, y1, color="red", linestyle="solid", label="High")
plt.plot(x, y2, color="blue", linestyle="dashed", label="Low")
plt.title("Line Plot", fontsize=20)
plt.xlabel("Day", fontsize=16)
plt.ylabel("Temperature", fontsize=16)
plt.legend()
plt.grid(color="red", linestyle="dotted")
plt.savefig("ch1ex12fig1.png")