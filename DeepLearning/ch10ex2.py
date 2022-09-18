# 1次元畳み込みニューラルネットワークの学習
# ＋学習曲線表示
# 入力データ：正弦波
# 出力データ：シフトした入力の和
# 活性化関数：tanh(x)

import numpy as np
import matplotlib.pyplot as plt
import datetime

# tanh関数の微分
def dtanh(x):
        return 1.0/(np.cosh(x)*np.cosh(x))

# データの用意
def preparedata(kk, nn):
    t = np.arange(kk)
    x = np.sin(t/100.0*np.pi)
    h=np.array([1.0, 0.8, 0.4])
    xx = np.zeros([kk, nn])
    for n in range(nn):
        xx[n:kk,n] = x[0:kk-n]
    y = np.dot(h, xx.T)
    ymax = np.max(np.abs(y))
    y = y/ymax
    one = np.ones([kk,1])
    onexx = np.concatenate((one, xx), 1)

    return(onexx, y)

# 重みの学習
def weightlearning(wwold, errork, xxk, yyk, eta):
    wwnew = wwold + eta*errork*xxk*dtanh(yyk)

    return wwnew

# 単出力パーセプトロン
def perceptron(ww, xxk):
    y = np.dot(ww,xxk)

    return y

# 誤差の評価
def evaluateerror(error, shiftlen, k):
    if(k>shiftlen):
        errorshift = error[k+1-shiftlen:k]
    else:
        errorshift = error[0:k]
    evalerror = np.sqrt(np.dot(errorshift, errorshift)/len(errorshift))

    return evalerror

# グラフを作成
def plotevalerror(evalerror, kk):
    x = np.arange(0, kk, 1)
    plt.figure(figsize=(10, 6))
    plt.plot(x, evalerror[0:kk])
    plt.title("Root Mean Square Error", fontsize=20)
    plt.xlabel("k", fontsize=16)
    plt.ylabel("RMSE", fontsize=16)
    plt.savefig("ch10ex2fig1.png")

    return

# メイン関数
def main():
    eta = 2.20
    shiftlen = 100
    epsilon = 1/shiftlen
    # データを用意
    kk = 5000
    nn = 3
    xx, zztrue = preparedata(kk, nn)
    kk, mm = xx.shape
    print("kk=", kk)
    print("mm=", mm)
    # 繰返し：学習過程
    wwold = np.zeros(mm)
    error = np.zeros(kk)
    evalerror = np.zeros(kk)
    for k in range(kk):
        yyk = perceptron(wwold, xx[k])
        zzk = np.tanh(yyk)
        error[k] = zztrue[k]-zzk
        evalerror[k] = evaluateerror(error, shiftlen, k)
        print("k={0}  zztrue={1:10.6f}  zz={2:10.6f}  RMSE={3:10.8f}".format(k, zztrue[k], zzk, evalerror[k]))
        if(k>shiftlen and evalerror[k]<epsilon):
            break
        wwnew = weightlearning(wwold, error[k], xx[k], yyk, eta)
        wwold = wwnew
    # 重みの学習結果を表示
    print("重みの学習結果:")
    for m in range(mm):
        print("w{0}={1:.8f}".format(m, wwold[m]))
    plotevalerror(evalerror, k)

    return

# ここから実行
if __name__ == "__main__":
	start_time = datetime.datetime.now()
	main()
	end_time = datetime.datetime.now()
	elapsed_time = end_time - start_time
	print("経過時間=", elapsed_time)
	print("すべて完了 !!!")