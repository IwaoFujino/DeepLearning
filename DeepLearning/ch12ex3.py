# 2次元畳み込みニューラルネットワークの学習
# ＋ 学習曲線表示
# 入力画像：face
# 出力画像：ソーベルフィルタによる輪郭抽出したface
# 活性化関数：ReLU
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import datetime

# ReLU関数
def relu(x):
    return  np.where(x<0, 0, x)

# ReLU関数の微分
def drelu(x):
    return  np.where(x<0, 0, 1)

# データの用意
def preparedata():
    # カーネル（ソーベルフィルタ）
    kx = np.array([[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
            ])
    # 入力画像を読み込む．
    imgin = misc.face(gray=True).astype(np.float32)
    # 2次元畳み込み演算により出力画像を計算
    print("kx=")
    print(kx)
    h = kx.flatten()
    nn = h.shape[0]
    # 2次元畳み込み演算により出力画像を計算
    kk1, kk2 = imgin.shape
    kk = kk1*kk2
    mm1, mm2 = kx.shape
    # mm1，mm2は奇数
    m1max = int((mm1-1)/2)
    m2max = int((mm2-1)/2)
    xx = np.zeros([kk1+mm1, kk2+mm2, mm1, mm2])
    for m1 in range(mm1):
        for m2 in range(mm2):
            xx[m1:kk1+m1, m2:kk2+m2, m1, m2] = imgin[0:kk1, 0:kk2]
    xxx = np.zeros([kk, nn])
    for k1 in range(kk1):
        for k2 in range(kk2):
            xxflip = np.flip(xx[k1+m1max, k2+m2max, :, :])
            k = k1*kk2 + k2
            xxx[k] = xxflip.flatten()
    y = np.dot(h, xxx.T)
    y = np.where(y<0, 0, y)
    xxxmax = np.max(np.abs(xxx))
    xxx = xxx/xxxmax
    ymax = np.max(np.abs(y))
    y = y/ymax
    xxxwith1 = np.ones([kk, nn+1])
    xxxwith1[:, 1:nn+1] = xxx[:, 0:nn]

    return(xxxwith1, y)

# 重みの学習
def weightlearning(wwold, errork, xxk, yyk, eta):
    wwnew = wwold + eta*errork*xxk*drelu(yyk)

    return wwnew

# 線形結合器
def linearcombiner(ww, xxk):
    y = np.dot(ww, xxk)

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
    plt.savefig("ch12ex3fig1.png")

    return

# メイン関数
def main():
    eta = 0.01
    shiftlen = 100
    epsilon = 1/shiftlen
    # データを用意
    xx, zztrue = preparedata()
    kk, mm = xx.shape
    print("kk=", kk)
    print("mm=", mm)
    # 繰返し：学習過程
    wwold = np.zeros(mm)
    error = np.zeros(kk)
    evalerror = np.zeros(kk)
    for k in range(kk):
        yyk = linearcombiner(wwold, xx[k])
        zzk = relu(yyk)
        error[k] = zztrue[k] - zzk
        evalerror[k] = evaluateerror(error, shiftlen, k)
        print("k={0}  zztrue={1:10.6f}  zz={2:10.6f}  evalerror={3:10.8f}".format(k,zztrue[k],zzk,evalerror[k]))
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
