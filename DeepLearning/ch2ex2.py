#ニュートン法により，2の平方根を求める.
kk = 10
xold = 3
for k in range(kk):
    delta = -(xold*xold-2)/(2*xold)
    xnew = xold + delta
    print("k=", k, "xnew=", xnew)
    # 新旧交代
    xold = xnew