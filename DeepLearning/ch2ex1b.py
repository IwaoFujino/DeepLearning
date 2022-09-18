# 総和s=1+2+3+...+kkを計算
# kk: キーボードから入力
kk = int(input("Input kk =?"))
sold = 0
for k in range(kk+1):
    snew = sold + k
    print("k=", k, "snew=", snew)
    # 新旧交代
    sold = snew