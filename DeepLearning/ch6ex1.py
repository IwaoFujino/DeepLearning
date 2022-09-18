# Irisデータの勾配降下法学習＋グラフ作成
# 活性化関数：ソフトマックス関数

import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn import datasets
from sklearn.utils import shuffle

np.set_printoptions(formatter={'float': '{:.4f}'.format})

# ソフトマックス関数
def softmax(x):
    s = np.exp(x)/np.sum(np.exp(x))

    return s

# データの用意
def preparedata():
    # データを読み込む．
    iris = datasets.load_iris()
    dataxx, datazz = shuffle(iris['data'],iris['target'], random_state=0)
    # dataxxの正規化
    xxmax = np.amax(dataxx)
    xx = dataxx/xxmax
    zzmax = np.amax(datazz)
    datazzonehot = np.zeros([len(datazz),zzmax+1])
    for k in range(len(datazz)):
        datazzonehot[k,datazz[k]] = 1
    kk, mm = xx.shape
    one = np.ones([kk,1])
    onexx = np.concatenate((one, xx), 1)

    return(onexx, datazzonehot)

# 重みの学習
def weightlearning(wwold, errork, xxk, yyk, eta):
    nn, mm = wwold.shape
    wwnew = np.empty([nn, mm])
    s = softmax(yyk)
    for n in range(nn):
        for m in range(mm):
            wwnew[n, m] = wwold[n, m] + eta*errork[n]*xxk[m]*s[n]*(1-s[n])

    return wwnew

# 線形結合器
def linearcombiner(ww, xxk):
    y = np.dot(ww,xxk)

    return y

# 誤差評価
def evaluateerror(error, shiftlen, k):
    ll, nn = error.shape
    errorshift = np.zeros([shiftlen,nn])
    if(k>=shiftlen):
        errorshift[0:shiftlen,0:nn] = error[k-shiftlen:k, 0:nn]
    else:
        errorshift[0:k,0:nn] = error[0:k, 0:nn]
    sqsumerror = np.empty(nn)
    for n in range(nn):
        sqsumerror[n] = np.dot(errorshift[:, n], errorshift[:,n])
    if(k>=shiftlen):
        evalerror = np.sqrt(np.sum(sqsumerror)/(shiftlen*nn))
    else:
        evalerror = np.sqrt(np.sum(sqsumerror)/((k+1)*nn))

    return evalerror

# グラフを作成
def plotevalerror(evalerror, kk):
    x = np.arange(0, kk, 1)
    plt.figure(figsize=(10, 6))
    plt.plot(x, evalerror[0:kk])
    plt.title("Root Mean Squared Error", fontsize=20)
    plt.xlabel("k", fontsize=16)
    plt.ylabel("RMSE", fontsize=16)
    plt.savefig("ch6ex1fig1.png")

    return

# メイン関数
def main():
    eta = 4.0
    shiftlen = 100
    epsilon = 1.0/(float(shiftlen))
    # データを用意する
    xx, zztrue = preparedata()
    kk, mm = xx.shape
    print("kk=", kk)
    print("mm=", mm)
    ll, nn = zztrue.shape
    print("ll=", ll)
    print("nn=", nn)
    wwold = np.zeros([nn, mm])
    error = np.zeros([kk, nn])
    evalerror = np.zeros(kk)
    ww = np.empty([kk, nn, mm])
    # 繰り返し：学習過程
    for k in range(kk):
        yyk = linearcombiner(wwold, xx[k])
        zzk = softmax(yyk)
        error[k] = zztrue[k] - zzk
        evalerror[k] = evaluateerror(error, shiftlen, k)
        print("k={0:4d} zztrue=[{1:.0f},{2:.0f},{3:.0f}] zz=[{4:.5f},{5:.5f},{6:.5f}] RMSE={7:.5f}".format(k, zztrue[k,0], zztrue[k,1], zztrue[k,2], zzk[0], zzk[1], zzk[2], evalerror[k]))
        if(k>shiftlen and evalerror[k]<epsilon):
            break
        wwnew = weightlearning(wwold, error[k], xx[k], yyk, eta)
        wwold = wwnew
        ww[k,:,:] = wwold
    # 重みの学習結果を表示
    print("重みの学習結果:")
    for m in range(nn):
        print("ww"+str(m)+"=", wwold[m, :])
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
