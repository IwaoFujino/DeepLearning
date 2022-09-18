# NumPyの基本：行列の結合
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
a3 = np.concatenate([a1, a2], axis=0)
print("a1とa2を縦に結合した結果：")
print(a3)
a4 = np.concatenate([a1, a2], axis=1)
print("a1とa2を横に結合した結果：")
print(a4)