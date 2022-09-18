# Kerasによる2次元畳み込みニューラルネットワークの学習
# ＋ 学習曲線表示
# データ：MNIST(手書き数字の識別)
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
# データを用意
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
xtrainorgshape = xtrain.shape
xtestorgshape = xtest.shape
print("shape of original xtrain =",xtrain.shape)
print("shape of original xtest =",xtest.shape)
xtrain = xtrain/np.max(xtrain)
xtrain = xtrain.reshape((xtrainorgshape[0], xtrainorgshape[1], xtrainorgshape[2], 1))
xtest = xtest/np.max(xtest)
xtest = xtest.reshape((xtestorgshape[0], xtestorgshape[1], xtestorgshape[2], 1))
print("shape of xtrain =",xtrain.shape)
print("shape of xtest =",xtest.shape)
nnlabel = 10
ytrain1hot = to_categorical(ytrain, nnlabel)
ytest1hot = to_categorical(ytest, nnlabel)
# モデルを構築
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),input_shape=(xtrainorgshape[1], xtrainorgshape[2], 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(nnlabel, activation="softmax"))
#model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# モデルの学習
history = model.fit(x=xtrain, y=ytrain1hot, epochs=50, verbose=1)
# 学習曲線を作成
plt.figure(figsize=(10, 6))
plt.plot(history.epoch, history.history["loss"])
plt.title("Learning Curve", fontsize=20)
plt.xlabel("Epoch")
plt.ylabel("Loss", fontsize=16)
plt.savefig("ch12ex4fig1.png")

plt.figure(figsize=(10, 6))
plt.plot(history.epoch, history.history["accuracy"])
plt.title("Learning Curve", fontsize=20)
plt.xlabel("Epoch")
plt.ylabel("Accuracy", fontsize=16)
plt.savefig("ch12ex4fig2.png")
