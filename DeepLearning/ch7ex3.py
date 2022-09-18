# ソフトマックス関数のオーバフロー対策
import numpy as np

# ソフトマックス関数
def softmax(x):
    xmax = np.max(x)
    sm = np.exp(x-xmax)/np.sum(np.exp(x-xmax))

    return sm
    
#メイン処理
x = np.array([6.0, 8.0])
for i in range(10):
    xx = 10.0*i*x
    print("i=", i, "xx=", xx, "softmax(xx)=", softmax(xx))