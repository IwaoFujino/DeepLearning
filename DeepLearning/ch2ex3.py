# 勾配降下法により，2の平方根を求める.
import numpy as np

kk = 2000
x = np.empty(kk+1)
x[0] = 3
eta = 0.001
for k in range(kk):
    delta = -4.0*x[k]*(2 - x[k]**2)
    x[k+1] = x[k] - eta*delta
    print("k=", k, "x=", x[k+1])