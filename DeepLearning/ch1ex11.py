# Matplotlibの基本：散布図とデータの値の表示
import numpy as np
import matplotlib.pyplot as plt

# データを用意
nn=5
x = np.arange(-nn, nn+1, 1)
y = x**2
# 散布図を作成
plt.figure(figsize=(10, 6))
plt.scatter(x, y, c="red")
for data in zip(x,y):
    plt.annotate(str(data), data)
plt.title("Scatter Plot", fontsize=20)
plt.xlabel("x", fontsize=16)
plt.ylabel("y", fontsize=16)
plt.savefig("ch1ex11fig1.png")
plt.show()
