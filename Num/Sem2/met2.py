import numpy as np


def func(x):
    x = np.array(x)
    return np.log(x) + 7/(2*x +6)


def interpol(X, Y):
    """
    this method should find polynomial interpolation
    :param X: X-values (1xN)
    :param Y: Y-values (1xN)
    :return: coefficients of N-1-degree polynome P (1xN)
    """
    mat = np.vander(X, len(X))
    P = np.linalg.solve(mat, Y)
    return P


