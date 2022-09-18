# 1出力2層ニューラルネットワークによるXORゲートの学習
import numpy as np
import matplotlib.pyplot as plt
import datetime

np.set_printoptions(formatter={'float': '{:.4f}'.format})

# シグモイド関数
def sigmoid(x):
    return 1/(1+np.exp(-x))

# ソフトマックス関数
def softmax(x):
    xmax = np.max(x)
    sm = np.exp(x-xmax)/np.sum(np.exp(x-xmax))

    return sm

# データの用意
def preparedata(datafilename):
    # データを読み込む
    data = np.load(datafilename)
    #print(data.files)
    # xxの正規化
    xxmax = np.amax(data["x"])
    xx = data["x"]/xxmax
    yy = data["y"]
    kk, mm = xx.shape
    one = np.ones([kk,1])
    onexx = np.concatenate((one,xx), 1)

    return onexx, yy

# 第1層（多出力）の重みの学習
def layer1weightlearning(wwold, deltak, xxk, uuk, eta):
    nn, mm = wwold.shape
    wwnew = np.empty([nn, mm])
    for n in range(nn):
        for m in range(mm):
            wwnew[n, m] = wwold[n, m] + eta*deltak[n]*uuk[n]*(1-uuk[n])*xxk[m]

    return wwnew

# 第2層（単出力）の重みの学習
def layer2weightlearning(wwold, deltak, xxk, zzk, eta):
    wwnew = wwold + eta*deltak*xxk*zzk*(1-zzk)

    return wwnew

def singleoutneuralnetwork(wwl1, wwl2, xxwith1k):
    yyk = np.dot(wwl1,xxwith1k)
    uuk = softmax(yyk)
    vvk = np.dot(wwl2,uuk)
    zzk = sigmoid(vvk)

    return yyk, uuk, vvk, zzk

# 誤差の評価(単出力)
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
    plt.title("Root Mean Squared Error", fontsize=20)
    plt.xlabel("k", fontsize=16)
    plt.ylabel("RMSE", fontsize=16)
    plt.savefig("ch7ex4fig1.png")

    return

# メイン関数
def main():
    #第1層の学習率
    eta1 = 1.0
    #第2層の学習率
    eta2 = 6.0
    #第1層出力数(第2層入力数)
    nn = 3
    shiftlen = 100
    epsilon = 1.0/(float(shiftlen))
    # データを用意
    xxwith1, tt = preparedata("./xorgate10000.npz")
    kk, mm = xxwith1.shape
    print("サンプル数=",kk)
    print("第1層入力数=",mm)
    print("第1層出力数(第2層入力数)=",nn)
    print("第2層出力数=",1)
    # 初期値の設定
    wwl1old = np.eye(nn, mm, dtype=float)
    wwl2old = np.ones(nn)
    deltal1k = np.zeros(nn)
    deltal2 = np.zeros(kk)
    evalerror = np.zeros(kk)
    # メイン繰返し：学習過程
    for k in range(kk):
        yyk, uuk, vvk, zzk = singleoutneuralnetwork(wwl1old, wwl2old, xxwith1[k])
        deltal2[k] = tt[k] - zzk
        for n in range(nn):
            deltal1k[n] = deltal2[k]*wwl2old[n]
        evalerror[k] = evaluateerror(deltal2, shiftlen, k)
        print("k={0:4d}   tt={1:.0f}   zz={2:.8f}    evalerror={3:.8f}".format(k, tt[k], zzk, evalerror[k]))
        if(k>shiftlen and evalerror[k]<epsilon):
            break
        wwl1new = layer1weightlearning(wwl1old, deltal1k, xxwith1[k], uuk, eta1)
        wwl2new = layer2weightlearning(wwl2old, deltal2[k], uuk, zzk, eta2)
        wwl1old = wwl1new
        wwl2old = wwl2new
    # 重みの学習結果を表示
    print("学習の結果:")
    print("第1層の重み")
    for n in range(nn):
        print(str(n)+"行=", wwl1old[n, :])
    print("第2層の重み")
    print(wwl2old)
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
