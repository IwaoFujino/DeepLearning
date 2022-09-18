# 中間試験と期末試験を乱数で生成，総合成績を作成
import numpy as np

# データの行数
kk = 10000
# データの列数
nn = 2
# データ作成
xx = np.random.randint(0,101, (kk, nn))
yy = 0.45*xx[:,0] + 0.55*xx[:,1]
# 作成したデータをnpzファイルに保存
np.savez("sougouseiseki.npz", x=xx, y=yy)