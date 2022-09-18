# NumPyの基本：乱数生成
import numpy as np

# 0〜1の一様分布の乱数
a1 = np.random.rand(5)
print("a1=")
print(a1)
# 整数型の一様分布の乱数
a2 = np.random.randint(1, 101, (3, 3))
print("a2=")
print(a2)
