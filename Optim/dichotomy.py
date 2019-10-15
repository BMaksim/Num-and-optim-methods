import numpy as np


def targetFunc(x):
    return 2*x + 5/np.exp(x)


a, b, eps, delta = -5, 5, 0.0001, 0.001

while (b - a)/2 >= eps:
    x1, x2 = (a + b - delta)/2, (a + b + delta)/2
    if targetFunc(x1) < targetFunc(x2):
        b = (b + a)/2
    else:
        a = (b + a)/2

print("В качестве минимума можно принять f(x) = {}, в точке x = {}".format(targetFunc((b + a)/2), (b + a)/2))

