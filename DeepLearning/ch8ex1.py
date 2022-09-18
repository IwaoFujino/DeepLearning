# digitsデータセットの学習 ＋ 学習曲線グラフ作成
# 構成：多層多クラスニューラルネットワーク
# 活性化関数：出力層はソフトマックス関数，それ以外はシグモイド関数
# 設定パラメータ：総層数，各層の学習率，各層（出力層除く）の出力端子数，
#（入力層の入力端子数と出力層の出力端子数はデータのサイズから自動対応）

import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn import datasets, metrics
from sklearn.utils import shuffle

# シグモイド関数
def sigmoid(x):
    return 1/(1+np.exp(-x))

# ソフトマックス関数
def softmax(x):
    xmax = np.max(x)
    sm = np.exp(x-xmax)/np.sum(np.exp(x-xmax))

    return sm

# データの用意
def preparedata():
    # データを読み込む．
    digits = datasets.load_digits()
    dataxx, datazz = shuffle(digits['data'], digits['target'], random_state=0)
    # dataxxの正規化
    xxmax = np.amax(dataxx)
    xx = dataxx/xxmax
    zzmax = np.amax(datazz)
    # one-hotベクトル
    datazzonehot = np.zeros([len(datazz),zzmax+1])
    for t in range(len(datazz)):
        datazzonehot[t,datazz[t]] = 1
    tt, mm = xx.shape
    one = np.ones([tt,1])
    onexx = np.concatenate((one,xx), 1)

    return onexx, datazzonehot

# 各層重みの初期化
def multilayerweightsinit(ll, mm, nn):
    wwold = {}
    for l in range(ll):
        wwoldtmp = np.eye(nn[l], mm[l], dtype=float)
        layer = "L"+str(l)
        wwold[layer] = wwoldtmp

    return wwold

# 多層誤差の逆伝播
def multilayerbackpropagation(ll, ww, errort):
    deltat = {}
    lastlayer = "L"+str(ll-1)
    deltat[lastlayer] = errort
    for l in range(ll-2, -1, -1):
        layer = "L"+str(l)
        nextlayer = "L"+str(l+1)
        deltat[layer] = np.dot(ww[nextlayer].T, deltat[nextlayer])

    return deltat

# 多層重みの学習
def multilayerweightslearning(ll, wwold, deltat, xxt, zzt, eta):
    wwnew = {}
    for l in range(ll):
        layer = "L"+str(l)
        wwoldtmp = wwold[layer]
        deltattmp = deltat[layer]
        xxttmp = xxt[layer]
        zzttmp = zzt[layer]
        nn, mm = wwoldtmp.shape
        wwnewtmp = np.empty([nn, mm])
        for n in range(nn):
            wwnewtmp[n, :] = wwoldtmp[n, :] + eta[l]*deltattmp[n]*zzttmp[n]*(1-zzttmp[n])*xxttmp[:]
        wwnew[layer] = wwnewtmp

    return wwnew

# 多層ニューラルネットワーク
def multilayerneuralnetwork(ll, wwold, xxwith1t):
    xxt = {}
    xxt["L0"] = xxwith1t
    yyt = {}
    zzt = {}
    for l in range(ll):
        layer = "L"+str(l)
        yyt[layer] = np.dot(wwold[layer], xxt[layer])
        if l==ll-1:
            zzt[layer] = softmax(yyt[layer])
        else:
            zzt[layer] = sigmoid(yyt[layer])
        nextlayer = "L"+str(l+1)
        xxt[nextlayer] = zzt[layer]

    return xxt, zzt

# 誤差評価
def evaluateerror(error, shiftlen, t):
    ll, nn = error.shape
    errorshift = np.zeros([shiftlen,nn])
    if(t>=shiftlen):
        errorshift[0:shiftlen, 0:nn] = error[t-shiftlen:t, 0:nn]
    else:
        errorshift[0:t,0:nn] = error[0:t, 0:nn]
    sqsumerror=np.empty(nn)
    for n in range(nn):
        sqsumerror[n] = np.dot(errorshift[:, n], errorshift[:, n])
    if(t>=shiftlen):
        evalerror = np.sqrt(np.sum(sqsumerror)/(shiftlen*nn))
    else:
        evalerror = np.sqrt(np.sum(sqsumerror)/((t+1)*nn))

    return evalerror

# グラフを作成する．
def plotevalerror(evalerror, tt):
    x = np.arange(0, tt, 1)
    plt.figure(figsize=(10, 6))
    plt.plot(x, evalerror[0:tt])
    plt.title("Root Mean Square Error", fontsize=20)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("RMSE", fontsize=16)
    plt.savefig("ch8ex1fig1.pdf")

    return

# メイン関数
def main():
    # 基本パラメータの設定
    ll = 3    # 総層数
    print("総層数=",ll)
    eta = [0.01, 0.05, 1.0]   # 各層の学習率
    shiftlen = 100    # 誤差の評価対象となる期間の長さ
    epsilon = 0.001  # 誤差評価量の許容値
    epochs = 200 # エポック数
    # データを用意する
    xxwith1, zztrue = preparedata()
    tt0, mm0 = xxwith1.shape
    tt0, nn0 = zztrue.shape
    print("データサンプル数=",tt0)
    tt = epochs*tt0
    mm = np.empty(ll, dtype=np.int16)
    nn = np.empty(ll, dtype=np.int16)
    # 各層の出力端子数
    nn = [10, 10, nn0]
    # 第1層の入力端子数
    mm[0] = mm0
    # 第2層以降の入力端子数
    for l in range(0, ll-1):
        mm[l+1] = nn[l]
    for l in range(ll):
        print("第{0}層：入力端子数={1} 出力端子数={2}".format(l, mm[l], nn[l]))
    # 重み初期値の設定
    wwold = multilayerweightsinit(ll, mm, nn)
    # 誤差の初期値
    error = np.zeros([tt, nn0])
    evalerror = np.zeros(tt)
    epochevalerror = np.zeros(epochs)
    breakflag = 0
    # メイン繰返し：学習過程
    # エポックの繰返し
    for epoch in range(epochs):
        # データを用意する
        xx, zztrue = preparedata()
        # データサンプルの繰返し
        zzprob = np.empty([tt0, nn0])
        for t0 in range(tt0):
            t = epoch*tt0 + t0
            # 信号の順伝播
            xxt0, zzt0 = multilayerneuralnetwork(ll, wwold, xxwith1[t0])
            lastlayer = "L"+str(ll-1)
            zzprob[t0] = zzt0[lastlayer]
            # 誤差の逆伝播
            error[t] = zztrue[t0]-zzt0[lastlayer]
            deltat = multilayerbackpropagation(ll, wwold, error[t])
            # 誤差の評価
            evalerror[t] = evaluateerror(error, shiftlen, t)
            if(t>shiftlen and evalerror[t]<epsilon):
                breakflag = 1
                break
            # 重みの更新
            wwnew = multilayerweightslearning(ll, wwold, deltat, xxt0, zzt0, eta)
            wwold = wwnew
        epochevalerror[epoch] = evalerror[t]
        print("epoch={0} RMSE={1:.8f}".format(epoch, epochevalerror[epoch]))
        if breakflag==1:
            break
    # 分類結果の評価
    truelabel = np.argmax(zztrue, axis=-1)
    zzlabel = np.argmax(zzprob, axis=-1)
    for true, zz in zip(truelabel, zzlabel):
        print("真ラベル=", true, " 推測ラベル=", zz, )
    print("分類結果の評価:")
    print("正解率 = {0:.6f}".format(metrics.accuracy_score(truelabel, zzlabel)))
    print(metrics.classification_report(truelabel, zzlabel))
    # 重みの学習結果を表示
    print("重みの学習結果:")
    for l in range(ll):
        layer = "L"+str(l)
        print("第{}層の重み".format(l))
        wwoldtmp = wwold[layer]
        for n in range(nn[l]):
            print(n,"行:", end="")
            for m in range(mm[l]):
                print("{0:8.4f}".format(wwoldtmp[n,m]), end="")
            print()
    # 学習曲線の作成と保存
    plotevalerror(epochevalerror, epoch)

    return

# ここから実行
if __name__ == "__main__":
	start_time = datetime.datetime.now()
	main()
	end_time = datetime.datetime.now()
	elapsed_time = end_time - start_time
	print("経過時間=", elapsed_time)
	print("すべて完了 !!!")
