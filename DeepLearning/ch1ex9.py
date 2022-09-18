# NumPyの基本：行列の絶対値，最大値
import numpy as np

# 行列の用意
a1 = np.array([[-4, 1, 1],
    [2, 2, 2],
    [3, 3, 3]])
print("a1=")
print(a1)
# 絶対値
a2 = np.abs(a1)
print("a1の絶対値=")
print(a2)
# 最大値
a3 = np.amax(a1)
print("a1の最大値=", a3)
# 絶対値の最大値
a4 = np.amax(np.abs(a1))
print("a1の絶対値の最大値=", a4)