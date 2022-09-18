# NumPyの基本：ベクトルの内積と行列の積を求めるdot()関数
import numpy as np

# ベクトルの内積
v1 = np.array([1, 2, 3])
print("v1=", v1)
v2 = np.array([1, 2, 3])
print("v2=", v2)
v3 = np.dot(v1, v2)
print("v1とv2の内積=", v3)

# 行列の積
a1 = np.array([[1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]])
print("a1=")
print(a1)
a2 = np.array([[2, 2, 2],
    [2, 2, 2],
    [2, 2, 2]])
print("a2=")
print(a2)
a3 = np.dot(a1, a2)
print("a1とa2の行列の積=")
print(a3)
