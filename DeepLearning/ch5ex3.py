# 総合評価の成績データにミニバッチ勾配降下法＋グラフ作成 
import numpy as np
import matplotlib.pyplot as plt
import datetime

# データの用意
def preparedata(datafilename):
    # データを読み込む
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

# 重みの学習(ミニバッチ勾配降下法)
def weightlearning_batch(wwold, errorb, xxb, yyb, eta):
    delta = 0.0
    for t in range(len(errorb)):
        s = sigmoid(yyb[t])
        delta = delta + errorb[t]*xxb[t]*s*(1-s)
    wwnew = wwold + eta*delta

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

# シグモイド関数
def sigmoid(y):
    return 1/(1+np.exp(-y))

# グラフを作成
def plotevalerror(evalerror, kk):
    x = np.arange(0, kk, 1)
    plt.figure(figsize=(10, 6))
    plt.plot(x, evalerror[0:kk])
    plt.title("Root Mean Squared Error", fontsize=20)
    plt.xlabel("Batch", fontsize=16)
    plt.ylabel("RMSE", fontsize=16)
    plt.savefig("ch5ex3fig1.png")

    return

# メイン関数
def main():
    # 基本パラメータの設定
    eta = 20
    shiftlen = 100
    epsilon = 0.05
    tt = 10
    # データを用意する
    xx, zztrue = preparedata("sougouhyouka.npz")
    kk, mm = xx.shape
    print("kk=", kk)
    print("mm=", mm)
    wwold = np.zeros(mm)
    error = np.zeros(kk)
    evalerror = np.zeros(kk)
    bb = int(kk/tt)
    xxb = np.empty([tt,mm])
    yyb = np.empty(tt)
    errorb = np.empty(tt)
    evalerrorbb = np.empty(bb)
    breakflag = 0
    # メイン繰り返し：学習過程    
    for b in range(bb):
        xxb = xx[b*tt:(b+1)*tt]
        for t in range(tt):
            k = b*tt+t
            yyb[t] = linearcombiner(wwold, xxb[t])
            zzbt = sigmoid(yyb[t])
            error[k] = zztrue[k] - zzbt
            evalerror[k] = evaluateerror(error, shiftlen, k)
            print("k={0} zztrue={1:.4f} zz={2:.4f} RMSE={3:.8f}".format(k, zztrue[k], zzbt, evalerror[k]))
            if(k>shiftlen and evalerror[k]<epsilon):
                breakflag = 1
                break
        errorb = error[b*tt:(b+1)*tt]
        evalerrorbb[b] = evalerror[k]
        print("Batch >>> b={0} RMSE={1:.8f}".format(b, evalerrorbb[b]))
        if breakflag == 1:
            break
        wwnew = weightlearning_batch(wwold, errorb, xxb, yyb, eta)
        wwold = wwnew
    # 重みの学習結果を表示
    print("重みの学習結果:")
    for m in range(mm):
        print("w{0}={1:.8f}".format(m, wwold[m]))
    plotevalerror(evalerrorbb, b)

    return

# ここから実行
if __name__ == "__main__":
	start_time = datetime.datetime.now()
	main()
	end_time = datetime.datetime.now()
	elapsed_time = end_time - start_time
	print("経過時間=", elapsed_time)
	print("すべて完了 !!!")
    