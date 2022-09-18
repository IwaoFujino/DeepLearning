# ソフトマックス関数のオーバフローチェック
import numpy as np

# ソフトマックス関数
def softmax(x):
    sm = np.exp(x)/np.sum(np.exp(x))

    return sm

# メイン処理
x = np.array([6.0, 8.0])
for i in range(10):
    xx = 10.0*i*x
    print("i=", i, "xx=", xx, "softmax(xx)=", softmax(xx))