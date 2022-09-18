# XORゲートの入出力データを作成
import numpy as np

# データの数（行数）
kk = 10000
# 入力の数（列数）
nn = 2
# データを作成
xx = np.random.randint(0,2, (kk, nn))
yy = xx[:, 0] ^ xx[:, 1]
# 作成したデータをnpzファイルに保存
filename = "xorgate"+str(kk)+".npz"
np.savez(filename, x=xx, y=yy)
