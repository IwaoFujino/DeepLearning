# Kerasによるirisデータセットの2層ニューラルネットワークの学習
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, RMSprop, Adam
from sklearn import datasets, preprocessing, metrics
from sklearn.utils import shuffle
from keras.utils import np_utils

# データを用意
iris = datasets.load_iris()
dataxx = iris['data']
truelabel = iris['target']
dataxx, truelabel = shuffle(dataxx, truelabel)   
X = preprocessing.scale(dataxx)
Y = np_utils.to_categorical(truelabel)
tt, mm = X.shape
tt, nn = Y.shape

# モデルを構築
model = Sequential()
model.add(Dense(input_dim=mm, units=10, use_bias=True, activation="sigmoid"))
model.add(Dense(units=nn, use_bias=False, activation="softmax"))
# 学習条件を設定(optimizerの設定の比較：1つのみ#を外して実行)
model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.1))
#model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=0.1))
#model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.1))

# モデルの学習
history = model.fit(X, Y, epochs=5000, batch_size=15)

# 学習の評価結果を表示
lossmin = 0.05
result = model.evaluate(X, Y, batch_size=1)
print("評価の結果：")
print("損失関数=", result)
if result < lossmin:
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
plt.plot(history.epoch, history.history["loss"])
plt.title("Learning Curve", fontsize=20)
plt.xlabel("Epoch")
plt.ylabel("Loss", fontsize=16)
plt.savefig("ch9ex3fig1.png")
