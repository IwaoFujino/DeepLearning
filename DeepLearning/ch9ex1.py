# KerasによるXORゲートの2層ニューラルネットワークの学習 + モデルを保存
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# データを用意
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# モデルを構築
model = Sequential()
model.add(Dense(input_dim=2, units=3, use_bias=True, activation="sigmoid"))
model.add(Dense(units=1, use_bias=False, activation="sigmoid"))
# 学習条件を設定(lossの設定の比較：1つのみ#を外して実行)
model.compile(loss="mse", optimizer=SGD(lr=0.1))
#model.compile(loss="mean_squared_error", optimizer=SGD(lr=0.1))
#model.compile(loss="mean_squared_logarithmic_error", optimizer=SGD(lr=0.5))
#model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.01))

# モデルの学習
history = model.fit(X, Y, epochs=5000, batch_size=1)

# 学習の評価結果を表示
lossmin = 0.01
loss = model.evaluate(X, Y, batch_size=1)
print("評価の結果：")
print("損失関数=", loss)
if loss < lossmin:
    print("学習がうまくできました．")
    # モデルを保存
    model.save("ch9ex1model.hdf5")
    # データから予測
    outputprob = model.predict(X)
    outputlabel = (outputprob > 0.5).astype("int32")
    for x, y, prob, label in zip(X, Y, outputprob, outputlabel):
        print("入力データ=", x, "真ラベル=", y, "出力確率=", prob, "出力ラベル=", label)
    # 重み配列を表示
    ww0 = model.layers[0].get_weights()
    print("第0層の重み")
    for w in ww0:
        print(w)
    ww1 = model.layers[1].get_weights()
    print("第1層の重み")
    for w in ww1:
        print(w)
else:
    print("学習がうまくできませんでした．")

# 学習曲線を表示
plt.figure(figsize=(10, 6))
plt.plot(history.epoch, history.history["loss"])
plt.title("Learning Curve", fontsize=20)
plt.xlabel("Epoch")
plt.ylabel("Loss", fontsize=16)
plt.savefig("ch9ex1fig1.png")
