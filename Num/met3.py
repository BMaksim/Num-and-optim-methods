import numpy as np
from enum import Enum
from numpy.polynomial.polynomial import polyval
from numpy.polynomial.legendre import legval, legvander


class ApproxType(Enum):
    algebraic = 0
    legendre = 1
    harmonic = 2


def func(x):
    """
    this method should implement VECTORIZED target function
    """
    return 3*x + np.cos(x + 1)


def approx(X0, Y0, X1, approx_type: ApproxType, dim):
    """
    this method should perform approximation on [-1; 1] interval
    :param X0: X-values (1 x N0)
    :param Y0: Y-values (1 x N0)
    :param X1: approximation points (1 x N1)
    :param approx_type:
        0 - algebraic polynomes (1, x, x^2, ...)
        1 - legendre polynomes
        2 - harmonic
    :param dim: dimension
    :return Y1: approximated Y-values (1 x N1)
    :return a: vector (1 x dim) of approximation coefficients
    :return P: (for approx_type 0 and 1) coefficients of approximation polynome P (1 x dim)
    """
    if approx_type is ApproxType.algebraic:
        Q = np.vander(X0, dim, increasing=True)
        mat = Q.T @ Q
        b = np.dot(Q.T, Y0)
        P = np.linalg.solve(mat, b)
        y = polyval(X1, P)
        return y, [], P
    if approx_type is ApproxType.legendre:
        Q = legvander(X0, dim - 1)
        mat = Q.T @ Q
        b = np.dot(Q.T, Y0)
        P = np.linalg.solve(mat, b)
        y = legval(X1, P)
        return y, [], P
    raise Exception(f'approximation of type {approx_type} not supported yet')

    


