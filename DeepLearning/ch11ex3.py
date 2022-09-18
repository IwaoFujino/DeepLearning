# Kerasによる再帰型ニューラルネットワークの学習
# ＋ 学習曲線表示
# 入力データ：2列の正弦波 ＋ ランダムノイズ
# 出力データ：各列y(k)=ax(k)+by(k-1)
# 活性化関数：tanh(x)
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.recurrent import SimpleRNN

# 関数：データの用意
def preparedata(kk, nn):
    time = np.arange(kk)
    x1 = np.sin(2.0*np.pi*time/100.0*10.0) + 0.2*np.random.rand(kk)
    x2 = np.sin(2.0*np.pi*time/100.0) + 0.5*np.random.rand(kk)
    xx = np.vstack([x1, x2]).T
    a = np.array([1.0, 0.95])
    b = np.array([0.5, 0.3])
    yy = np.empty([kk, nn])
    yy[0] = np.dot(a, xx[0])
    for k in range(1, kk):
        yy[k] = np.dot(a, xx[k]) + np.dot(b, yy[k-1])
    yy0max =np.amax(np.abs(yy[:,0]))
    yy[:,0] =  yy[:,0]/yy0max
    yy1max = np.amax(np.abs(yy[:,1]))
    yy[:,1] =  yy[:,1]/yy1max

    return(xx, yy)

# データを用意
kk = 2000
nn = 2
shift = 100
x, y = preparedata(kk, nn)
xx = np.empty([kk-shift, shift, nn])
yy = np.empty([kk-shift, nn])
for k in range(0, kk-shift):
    xx[k, :, :] = x[k:k+shift, :]
    yy[k, :] = y[k+shift, :]
print("x shape =", xx.shape)
print("y shape =", yy.shape)

# モデルを構築
model = Sequential()
model.add(SimpleRNN(nn, input_shape=(shift, nn), activation='tanh'))
model.compile(loss='mse', optimizer='sgd')
model.summary()

# モデルの学習
history = model.fit(xx, yy, batch_size=10, epochs=50)

# 学習曲線を作成
plt.figure(figsize=(10, 6))
plt.plot(history.epoch, history.history["loss"])
plt.title("Learning Curve", fontsize=20)
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.savefig("ch11ex3fig1.png")
