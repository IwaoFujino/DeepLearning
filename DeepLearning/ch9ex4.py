# Kerasによるdigitsデータセットの3層ニューラルネットワークによる学習
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from sklearn import datasets, preprocessing, metrics
from sklearn.utils import shuffle
from keras.utils import np_utils

# データを用意
digits = datasets.load_digits()
dataxx = digits['data']
truelabel = digits['target']
dataxx, truelabel = shuffle(dataxx, truelabel)
X = preprocessing.scale(dataxx)
Y = np_utils.to_categorical(truelabel)
tt, mm = X.shape
tt, nn = Y.shape

# モデルを構築
model = Sequential()
model.add(Dense(input_dim=mm, units=30, use_bias=True, activation="sigmoid"))
model.add(Dense(units=20, use_bias=False, activation="sigmoid"))
model.add(Dense(units=nn, use_bias=False, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.1), metrics=['mse', 'accuracy'])

# モデルの学習
history = model.fit(X, Y, epochs=5000, batch_size=30)

# 学習の評価結果を表示
lossmin = 0.01
result = model.evaluate(X, Y, batch_size=1)
print("評価の結果：")
print("損失関数=", result[0])
if result[0] < lossmin:
    print("学習がうまくできました．")
    # データから予測
    outputprob = model.predict(X)
    outputlabel = np.argmax(model.predict(X), axis=-1)
    for true, output in zip(truelabel, outputlabel):
        print("真ラベル=", true, "  推測ラベル=", output)
    # 分類結果の評価
    print("分類結果の評価:")
    print("正解率 = {0:.6f}".format(metrics.accuracy_score(truelabel, outputlabel)))
    print(metrics.classification_report(truelabel, outputlabel))
else:
    print("学習がうまくできませんでした．")

# 学習曲線を表示
plt.figure(figsize=(10, 6))
plt.plot(history.epoch, history.history["mse"])
plt.title("Learning Curve", fontsize=20)
plt.xlabel("Epoch")
plt.ylabel("MSE", fontsize=16)
plt.savefig("ch9ex4fig1.png")

plt.figure(figsize=(10, 6))
plt.plot(history.epoch, history.history["accuracy"])
plt.title("Learning Curve", fontsize=20)
plt.xlabel("Epoch")
plt.ylabel("Accuracy", fontsize=16)
plt.savefig("ch9ex4fig2.png")
