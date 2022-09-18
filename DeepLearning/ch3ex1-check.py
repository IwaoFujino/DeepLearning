# sougouseiseki.npzの内容を確認
import numpy as np

# ファイルからデータをロード
seisekidata = np.load("sougouseiseki.npz")
print(seisekidata.files)
xx = seisekidata["x"]
yy = seisekidata["y"]
# xx, yyの内容を表示
for x, y in zip(xx,yy):
    print("{0:3d}  {1:3d}  {2:10.6f}".format(x[0], x[1], y))
# xx, yyのサイズ
print("xxのサイズ =", xx.shape)
print("yyのサイズ =", yy.shape)
