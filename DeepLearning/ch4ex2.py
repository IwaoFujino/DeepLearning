# ANDゲートの誤り訂正学習
import numpy as np
import datetime

# 重みの学習
def weightlearning(wwold, errork, xxk, eta):
    wwnew = wwold + eta*errork*xxk

    return wwnew

# 線形結合器
def perceptron(ww, xxk):
    y = np.dot(ww,xxk)

    return y

# 平均誤り
def checkerrorrate(error, shiftlen, k):
    if(k>shiftlen):
        errorshift = np.abs(error[k+1-shiftlen:k])
    else:
        errorshift = np.abs(error[0:k])
    errorave = np.average(errorshift)

    return errorave

# ステップ関数
def stepfunction(x):
    if x>=0:
        return 1
    else:
        return 0

# メイン関数
def main():
    eta = 5.0e-1
    epsilon = 0.001
    shiftlen = 100
    # データを読み込む．
    andgatedata = np.load("andgate10000.npz")
    xx = andgatedata["x"]
    kk, nn = xx.shape
    one = np.ones([kk,1])
    xx = np.concatenate((one, xx), 1)
    kk, nn = xx.shape
    zztrue = andgatedata["y"]
    print("zztrue size=", zztrue.shape)
    # 繰返し：学習過程
    wwold = [0.0, 0.0, 0.0]
    error = np.zeros(kk)
    errorave = np.zeros(kk)
    for k in range(kk):
        yyk = perceptron(wwold, xx[k])
        zzk = stepfunction(yyk)
        error[k] = zztrue[k] - zzk
        errorave[k] = checkerrorrate(error, shiftlen, k)
        print("k={0} zztrue={1:.4f} zz={2:.4f} errorave={3:.8f}".format(k,zztrue[k],zzk,errorave[k]))
        if(k>shiftlen and errorave[k]<epsilon):
            break
        wwnew = weightlearning(wwold, error[k], xx[k], eta)
        wwold = wwnew
    # 重みの学習結果を表示
    print("重みの学習結果: w0=", wwold[0], "w1=", wwold[1], "w2=", wwold[2])

    return

# ここから実行
if __name__ == "__main__":
	start_time = datetime.datetime.now()
	main()
	end_time = datetime.datetime.now()
	elapsed_time = end_time-start_time
	print("経過時間=", elapsed_time)
	print("すべて完了 !!!")
