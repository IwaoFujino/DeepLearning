# KerasによるXORゲートの2層ニューラルネットワークの学習 + モデルを復元して評価
import numpy as np
from keras.models import load_model

# データを用意
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# モデルを復元
model = load_model("ch9ex1model.hdf5")

# 学習の評価結果を表示
lossmin = 0.01
loss = model.evaluate(X, Y, batch_size=1)
print("評価の結果：")
print("損失関数=", loss)
if loss < lossmin:
    print("学習がうまくできました．")
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