# 総和s=1+2+3+...+kkを計算
# kk: キーボードから入力
kk = int(input("Input kk =?"))
s = 0
for k in range(kk+1):
    s = s + k
    print("k=", k, "s=", s)
