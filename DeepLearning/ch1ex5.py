# NumPyの基本：行列の要素の演算（加算を例として）
import numpy as np

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
a3 = a1 + a2
print("行列の要素の和a1+a2=")
print(a3)
