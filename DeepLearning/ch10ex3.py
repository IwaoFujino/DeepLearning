# Kerasによる1次元畳み込みニューラルネットワークの学習
# ＋学習曲線表示
# 入力データ：正弦波
# 出力データ：シフトした入力の和
# 活性化関数：tanh(x)
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D
# 学習のためのデータを用意
kk = 5000
nn = 3
t = np.arange(kk)
x = np.sin(t/100.0*np.pi)
h = np.array([1.0, 0.8, 0.4])
# 畳み込み演算でフィルタの出力を計算
xx = np.zeros([kk,nn])
for n in range(nn):
    xx[n:kk,n] = x[0:kk-n]
y = np.dot(h, xx.T)
# 入力データと出力データを用意
inputdata  = x
ymax = np.max(np.abs(y))
outputdata = y/ymax
inputdata = inputdata.reshape((-1, 1, 1))
outputdata = outputdata.reshape((-1, 1))
# モデルを構築する．
model = Sequential()
model.add(Conv1D(filters=1, input_shape=(1, 1), kernel_size=16, padding='same', activation='tanh'))
model.add(GlobalMaxPooling1D())
model.compile(loss='mse', optimizer='sgd')
#model.compile(loss='mse', optimizer='adam')
#model.compile(loss='mse', optimizer='rmsprop')
model.summary()
# モデルの学習
history = model.fit(x=inputdata, y=outputdata, epochs=50, verbose=1)
print("重みの学習結果：")
# 重み配列を表示
ww0 = model.layers[0].get_weights()
print("第0層の重み：")
print("バイアスの値={0:.8f}".format(ww0[1][0]))
print("カーネルの重み=")
for i in range(len(ww0[0])):
    print("w[{0:2d}]={1:12.8f}".format(i, ww0[0][i][0][0]))
# 学習曲線を作成
plt.figure(figsize=(10, 6))
plt.plot(history.epoch, history.history["loss"])
plt.title("Learning Curve", fontsize=20)
plt.xlabel("Epoch")
plt.ylabel("Loss", fontsize=16)
plt.savefig("ch10ex3fig1.png")
