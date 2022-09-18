# NumPyの基本：配列の形を変える.
import numpy as np

# 1行に表示する文字数を設定
np.set_printoptions(linewidth=70)
# 行列の用意
print("行列を用意する.")
a1 = np.arange(1, 61, 1)
print("a1=")
print(a1)
# 2次元配列に形を変える．
print("a1を2次元配列に形を変える．")
a2 = a1.reshape([6,10])
print("a2=")
print(a2)
# 2次元配列に形を変える．
print("a1を3次元配列に形を変える．")
a3 = a1.reshape([-1, 3, 5])
print("a3=")
print(a3)