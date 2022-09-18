# RNNの確率勾配降下法学習 ＋ 学習曲線表示
# 入力データ：正弦波+ノイズ
# 出力データ：y(k)=ax(k)+by(k-1)
# 活性化関数：tanh(x)
import numpy as np
import matplotlib.pyplot as plt
import datetime

np.set_printoptions(formatter={'float': '{:.6f}'.format})

# tanh関数の微分
def dtanh(x):
    return 1.0/(np.cosh(x)*np.cosh(x))

# データの用意
def preparedata(kk, nn):
    time = np.arange(kk)
    x1 = np.sin(2.0*np.pi*time/100.0*10.0) + 0.2*np.random.rand(kk)
    x2 = np.sin(2.0*np.pi*time/100.0) + 0.5*np.random.rand(kk)
    xx = np.vstack([x1, x2]).T
    a = np.array([1.0, 0.95])
    b = np.array([0.5, 0.3])
    yy = np.empty([kk, nn])
    for i in range(nn):
        yy[0, i] = 0.0
        for k in range(1, kk):
            yy[k, i] = a[i]*xx[k, i] + b[i]*yy[k-1, i]
    yy0max = np.amax(np.abs(yy[:,0]))
    yy[:,0] =  yy[:,0]/yy0max
    yy1max = np.amax(np.abs(yy[:,1]))
    yy[:,1] =  yy[:,1]/yy1max
    one = np.ones([kk,1])
    onexx = np.concatenate((one,xx), 1)

    return(onexx, yy)

# 重みwwの学習
def weightwwlearning(wwold, errork, xxk, yyk, etaw):
    nn, mm = wwold.shape
    wwnew = np.empty([nn, mm])
    for n in range(nn):
        for m in range(mm):
            wwnew[n, m] = wwold[n, m] + etaw*errork[n]*xxk[m]*dtanh(yyk[n])

    return wwnew

# 重みvvの学習
def weightvvlearning(vvold, errork, zzprev, yyk, etav):
    nn, mm = vvold.shape
    vvnew = np.empty([nn, mm])
    for n in range(nn):
        for m in range(mm):
            vvnew[n, m] = vvold[n, m] + etav*errork[n]*zzprev[m]*dtanh(yyk[n])

    return vvnew

# 再帰型システム
def recurrentsys(ww, xxk, vv, zzprev):
    y = np.dot(ww,xxk) + np.dot(vv,zzprev)

    return y

# 誤差評価
def checkerrorrate(error, shiftlen, k):
    ll, nn = error.shape
    errorshift=np.zeros([shiftlen,nn])
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
    plt.xlabel("k", fontsize=16)
    plt.ylabel("RMSE", fontsize=16)
    plt.savefig("ch11ex2fig1.png")

    return

# メイン関数
def main():
    etaw = 0.4
    etav = 0.5
    shiftlen = 100
    epsilon = 1.0/(float(shiftlen))
    # データを用意
    kk = 5000
    nn = 2
    xx, zztrue = preparedata(kk, nn)
    kk, nn = xx.shape
    print("kk=", kk)
    print("nn=", nn)
    ll, mm = zztrue.shape
    print("ll=", ll)
    print("mm=", mm)
    # 初期値の設定
    wwold = np.zeros([mm, nn])
    if(nn<mm):
        wwold[0:nn, 0:nn] = np.eye(nn)
    else:
        wwold[0:mm, 0:mm] = np.eye(mm)
    vvold = np.eye(mm)
    zzold = np.zeros(mm)
    error = np.zeros([kk, mm])
    evalerror = np.zeros(kk)
    # メイン繰返し
    for k in range(kk):
        yyk = recurrentsys(wwold, xx[k], vvold, zzold)
        zzk = np.tanh(yyk)
        error[k] = zztrue[k]-zzk
        evalerror[k] = checkerrorrate(error, shiftlen, k)
        print("k={0:5d}  zz=[{1:9.6f},{2:9.6f}]  RMSE={3:9.6f}".format(k, zzk[0], zzk[1], evalerror[k]))
        if(k>shiftlen and evalerror[k]<epsilon):
            break
        wwnew = weightwwlearning(wwold, error[k], xx[k], yyk, etaw)
        vvnew = weightvvlearning(vvold, error[k], zzold, yyk, etav)
        wwold = wwnew
        vvold = vvnew
        zzold = zzk
    # 重みの学習結果を表示
    print("重みの学習結果:")
    for m in range(mm):
        print("ww"+str(m)+"=", wwold[m, :])
    for m in range(mm):
        print("vv"+str(m)+"=", vvold[m, :])
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
