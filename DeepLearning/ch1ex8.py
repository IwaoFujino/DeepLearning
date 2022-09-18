# NumPyの基本：行列の指数関数
import numpy as np

# 行列の用意
a1 = np.array([[1, 1, 1],
    [2, 2, 2],
    [3, 3, 3]])
print("a1=")
print(a1)
# 指数関数
a2 = np.exp(-a1)
print("指数関数exp(-a1)=")
print(a2)

