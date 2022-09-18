# Irisデータの勾配降下法学習（ソフトマックス関数）) ＋ グラフ作成
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

# ソフトマックス関数の微分
def dsoftmax(x):
    s = softmax(x)
    ds = s*(1-s)

    return ds

# データの用意
def preparedata():
    # データを読み込む．
    iris = datasets.load_iris()
    dataxx, datazz = shuffle(iris['data'],iris['target'], random_state=0)
    # dataxxの正規化
    xxmax = np.amax(dataxx)
    xx = dataxx/xxmax
    zzmax=np.amax(datazz)
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
    dsk = dsoftmax(yyk)
    for n in range(nn):
        for m in range(mm):
            wwnew[n, m] = wwold[n, m]+eta*errork[n]*xxk[m]*dsk[n]

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
    plt.title("Root Mean Square Error", fontsize=20)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("RMSE", fontsize=16)
    plt.savefig("ch6ex2fig1.png")

    return

# メイン関数
def main():
    eta = 4.0
    shiftlen = 100
    epsilon = 1.0/(float(shiftlen))
    epochs = 500
    # データのサイズをチェック
    xx, zztrue = preparedata()
    kk0, mm = xx.shape
    kk = epochs*kk0
    print("kk=", kk)
    print("mm=", mm)
    ll, nn = zztrue.shape
    print("ll=", ll)
    print("nn=", nn)
    wwold = np.zeros([nn, mm])
    error = np.zeros([kk, nn])
    evalerror = np.zeros(kk)
    epocherror = np.zeros(epochs)
    lastwwold = np.empty([nn, mm])
    breakflag = 0 # 繰返し中止のフラグ
    # メイン繰返し：学習過程
    # エポックの繰返し
    for epoch in range(epochs):
          # データを用意
        xx, zztrue = preparedata()
        # データサンプルの繰返し
        for k0 in range(kk0):
            k = epoch*kk0+k0
            yyk0 = linearcombiner(wwold, xx[k0])
            zzk0 = softmax(yyk0)
            error[k] = zztrue[k0] - zzk0
            evalerror[k] = evaluateerror(error, shiftlen, k)
            lastwwold = wwold
            if(k>shiftlen and evalerror[k]<epsilon):
                breakflag = 1
                break
            wwnew = weightlearning(wwold, error[k], xx[k0], yyk0, eta)
            wwold = wwnew
        epocherror[epoch] = evalerror[k]
        print("epoch={0} RMSE={1:.8f}".format(epoch, epocherror[epoch]))
        if breakflag==1:
            break
    # 重みの学習結果を表示
    print("重みの学習結果:")
    for n in range(nn):
        print(str(n)+"番目出力：ww=", lastwwold[n, :])
    plotevalerror(epocherror, epoch)

    return

# ここから実行
if __name__ == "__main__":
	start_time = datetime.datetime.now()
	main()
	end_time = datetime.datetime.now()
	elapsed_time = end_time - start_time
	print("経過時間=", elapsed_time)
	print("すべて完了 !!!")
