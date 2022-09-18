# NumPyの基本：0値配列，1値配列と単位行列の生成
import numpy as np

# 2次元0値配列
a1 = np.zeros((3, 4))
print("2次元の0値配列：")
print(a1)
# 2次元1値配列
a2 = np.ones((3, 4))
print("2次元の1値配列：")
print(a2)
# 単位行列
a3 = np.eye(3)
print("単位行列：")
print(a3)
