# 総合評価の成績データに確率勾配降下法 ＋ グラフ作成
import numpy as np
import matplotlib.pyplot as plt
import datetime

# シグモイド関数
def sigmoid(x):
    s = 1/(1 + np.exp(-x))

    return s

# データの用意
def preparedata(datafilename):
    # データを読み込む．
    data = np.load(datafilename)
    #print(data.files)
    # データxxの正規化
    xxmax = np.amax(data["x"])
    xx = data["x"]/xxmax
    zztrue = data["y"]
    kk, mm = xx.shape
    one = np.ones([kk,1])
    onexx = np.concatenate((one, xx), 1)

    return(onexx, zztrue)

# 重みの学習
def weightlearning(wwold, errork, xxk, yyk, eta):
    s = sigmoid(yyk)
    wwnew = wwold + eta*errork*xxk*s*(1-s)

    return wwnew

# 線形結合器
def linearcombiner(ww, xxk):
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
    plt.savefig("ch5ex2fig1.pdf")

    return

# メイン関数
def main():
    eta = 30.0
    shiftlen = 100
    epsilon = 0.05
    # データを用意
    xx, zztrue = preparedata("sougouhyouka.npz")
    kk, mm = xx.shape
    print("kk=", kk)
    print("mm=", mm)
    wwold = np.zeros(mm)
    error = np.zeros(kk)
    evalerror = np.zeros(kk)
    # 繰返し：学習過程
    for k in range(kk):
        yyk = linearcombiner(wwold, xx[k])
        zzk = sigmoid(yyk)
        error[k] = zztrue[k] - zzk
        evalerror[k] = evaluateerror(error, shiftlen, k)
        print("k={0} zztrue={1:.4f} zz={2:.4f} RMSE={3:.8f}".format(k, zztrue[k], zzk, evalerror[k]))
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
