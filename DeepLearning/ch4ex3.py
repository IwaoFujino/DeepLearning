# ANDゲートの誤り訂正学習 ＋ グラフ作成
import numpy as np
import matplotlib.pyplot as plt
import datetime

# 重みの学習
def weightlearning(wwold, errork, xxk, eta):
    wwnew = wwold + eta*errork*xxk

    return wwnew

# 線形結合器
def linearcombiner(ww, xxk):
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

# 平均誤りのグラフを作成
def plotevalerror(errorave, kk):
    x = np.arange(0, kk, 1)
    plt.figure(figsize=(10, 6))
    plt.plot(x, errorave[0:kk])
    plt.title("Average Error", fontsize=20)
    plt.xlabel("k", fontsize=16)
    plt.ylabel("Average error", fontsize=16)
    plt.savefig("ch4ex3fig1.png")

    return

# 重みのグラフを作成
def plotweights(ww0, ww1, ww2, kk):
    x = np.arange(0, kk, 1)
    plt.figure(figsize=(10, 6))
    plt.plot(x, ww0[0:kk], color="red", linestyle="-", label="ww0")
    plt.plot(x, ww1[0:kk], color="blue", linestyle="--", label="ww1")
    plt.plot(x, ww2[0:kk], color="green", linestyle="-.", label="ww2")
    plt.title("Weights", fontsize=20)
    plt.xlabel("k", fontsize=16)
    plt.ylabel("Weight", fontsize=16)
    plt.legend()
    plt.savefig("ch4ex3fig2.png")

    return

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
    ww = np.empty([kk,nn])
    for k in range(kk):
        yyk = linearcombiner(wwold, xx[k])
        zzk = stepfunction(yyk)
        error[k] = zztrue[k] - zzk
        errorave[k] = checkerrorrate(error, shiftlen, k)
        print("k={0} zztrue={1:.4f} zz={2:.4f} errorave={3:.8f}".format(k, zztrue[k], zzk, errorave[k]))
        if(k>shiftlen and errorave[k]<epsilon):
            break
        wwnew = weightlearning(wwold, error[k], xx[k], eta)
        wwold = wwnew
        ww[k,:] = wwold
    # 重みの学習結果を表示
    print("重みの学習結果: w0=", wwold[0], "w1=", wwold[1], "w2=", wwold[2])
    plotevalerror(errorave, k)
    plotweights(ww[:,0], ww[:,1], ww[:,2], k)

    return

# ここから実行
if __name__ == "__main__":
	start_time = datetime.datetime.now()
	main()
	end_time = datetime.datetime.now()
	elapsed_time = end_time - start_time
	print("経過時間=", elapsed_time)
	print("すべて完了 !!!")
