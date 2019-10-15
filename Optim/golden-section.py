import numpy as np


def targetFunc(x):
    return 2*x + 5/np.exp(x)


a, b, eps, phi = -5, 5, 0.0001, (1+np.sqrt(5))/2
x1, x2 = b - (b - a)/phi, a + (b - a)/phi

while (b - a)/2 >= eps:
    if targetFunc(x1) < targetFunc(x2):
        b, x2, x1 = x2, x1, a + (b - x2)
    else:
        a, x1, x2 = x1, x2, b - (x1 - a)
print("В качестве минимума можно принять f(x) = {}, в точке x = {}".format(targetFunc((b + a)/2), (b + a)/2))
