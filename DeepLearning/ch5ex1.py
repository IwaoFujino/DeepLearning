# 中間試験と期末試験を乱数で生成，総合評価を作成
import numpy as np

# データの行数
kk = 10000
# データの列数
nn = 2
# データを作成
xx = np.random.randint(0,101, (kk, nn))
yy = 0.45*xx[:,0] + 0.55*xx[:,1]
zz = np.where(yy>=60, 1, 0)
# 作成したデータをnpzファイルに保存
np.savez("sougouhyouka.npz", x=xx, y=zz)