import numpy as np
from enum import Enum
from numpy.polynomial.polynomial import polyval
from numpy.polynomial.legendre import legval, legvander

def legandr(x, n):
        if n == 0:
            return 1
        if n == 1:
            return x
        else:
            return ((2 * n - 1) / (n)) * x * legandr(x, n - 1) - ((n - 1) / n) * legandr(x, n - 2)



class ApproxType(Enum):
    algebraic = 0
    legendre = 1
    harmonic = 2


def func(x):
    """
    this method should implement VECTORIZED target function
    """
    return abs(3*x + np.sin(x))


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
        b = Q.T @ Y0
        P = np.linalg.solve(mat, b)
        y = polyval(X1, P)
        return y, [], P
    if approx_type is ApproxType.legendre:
        Q = legvander(X0, dim - 1)
        mat = Q.T @ Q
        b = Q.T @ Y0
        P = np.linalg.solve(mat, b)
        y = legval(X1, P)

        D = [[0 for i in range(dim)] for j in range(dim)]
        D[0][0] = 1
        D[0][1] = 0
        D[1][0] = 0                                                               
        D[1][1] = 1                                                             
        i = 2                                  
        while i < (dim):
            D[i][0] = -(i-1)/(i)*D[i-2][0]  
            j = 0
            while j <= i:
                D[i][j] = (2*i-1)/i*D[i-1][j-1] - (i-1)/i*D[i-2][j]
                j+=1
            i+=1    
        D = np.array(D)
        G = P @ D

        return y, P, G
    raise Exception(f'approximation of type {approx_type} not supported yet')

    


