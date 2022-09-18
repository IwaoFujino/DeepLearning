# NumPyの基本：行列の転置，ベクトルと転置行列の積
import numpy as np

# ベクトル
h = np.array([1, 2, 3])
print("h=", h)
# 行列
xx = np.array([[1, 1, 1],
    [2, 2, 2],
    [3, 3, 3]])
print("xx=")
print(xx)
# 行列の転置
print("xxの転置=")
print(xx.T)
# ベクトルと転置行列の積
y = np.dot(h, xx.T)
print("hとxxの転置行列の積=")
print(y)
