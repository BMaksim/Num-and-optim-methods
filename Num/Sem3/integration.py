import numpy as np
from scipy.special import binom
import math

MAXITER = 12

def moments(max_s: int, xl: float, xr: float, a: float = 0.0, b: float = 1.0, alpha: float = 0.0, beta: float = 0.0):
    """
   compute moments of the weight 1 / (x-a)^alpha / (b-x)^beta over [xl, xr]
   max_s : highest required order
   xl : left limit
   xr : right limit
   a : weight parameter a
   b : weight parameter b
   alpha : weight parameter alpha
   beta : weight parameter beta
   """
 
    assert alpha * beta == 0, \
        f'alpha ({alpha}) and/or beta ({beta}) must be 0'
    moms = []
    if alpha != 0.0:
        assert a is not None, f'"a" not specified while alpha != 0'
        for j in range(max_s + 1):
            mom = ((xr - a) ** (j - alpha + 1) - (xl - a) ** (j - alpha + 1)) / (j - alpha + 1)
            for i, momi in enumerate(moms):
                mom += binom(j, i) * -((-a) ** (j-i)) * momi
            moms.append(mom)
        return np.array(moms)
    if beta != 0.0:
        assert b is not None, f'"b" not specified while beta != 0'
        for j in range(max_s + 1):
            mom = (-1) ** j * ((b - xl) ** (j - beta + 1) - (b - xr) ** (j - beta + 1)) / (j - beta + 1)
            for i, momi in enumerate(moms):
                mom -= binom(j, i) * ((-b) ** (j - i)) * momi
            moms.append(mom)
        return np.array(moms)
 
    if alpha == 0 and beta == 0:
        return np.array([(xr ** s - xl ** s) / s for s in range(1, max_s + 2)])
 
    mu = np.zeros(max_s + 1)
    return mu

def quad(f, xl: float, xr: float, nodes, *params):
    """
    small Newton—Cotes formula
    f: function to integrate
    xl : left limit
    xr : right limit
    nodes: nodes within [xl, xr]
    *params: parameters of the variant — a, b, alpha, beta)
    """
    mu = np.array(moments(len(nodes) - 1, xl, xr, *params))
    X = np.array([[nodes[j]**i for j in range(len(nodes))] for i in range(len(nodes))])
    quad_coefs = np.linalg.solve(X, mu)
    result = [coef*f(num) for coef, num in zip(quad_coefs, nodes)]
    result = np.sum(np.array(result))
    return result 


def quad_gauss(f, xl: float, xr: float, n: int, *params):
    """
    small Gauss formula
    f: function to integrate
    xl : left limit
    xr : right limit
    n : number of nodes
    *params: parameters of the variant — a, b, alpha, beta)
    """
    mu = np.array(moments(2 * n - 1, xl, xr, *params))
    mu_free = -mu[n:]
    M = np.array([[mu[i + j] for j in range(n)] for i in range(n)])
    a = np.linalg.solve(M, mu_free)
    a = a[::-1]
    a = np.append([1, ], a)
    xs = np.roots(a)
    X = np.array([[xs[j]**i for j in range(n)] for i in range(n)])
    gaus_coefs = np.linalg.solve(X, mu[:n])
    result = [coef*f(num) for coef, num in zip(gaus_coefs, xs)]
    result = np.sum(np.array(result))
    return result


def composite_quad(f, xl: float, xr: float, N: int, n: int, *params):
    """
    composite Newton—Cotes formula
    f: function to integrate
    xl : left limit
    xr : right limit
    N : number of steps
    n : number of nodes od small formulae
    *params: parameters of the variant — a, b, alpha, beta)
    """
    mesh = np.linspace(xl, xr, N + 1)
    return sum(quad(f, mesh[i], mesh[i + 1], equidist(n, mesh[i], mesh[i + 1]), *params) for i in range(N))


def composite_gauss(f, a: float, b: float, N: int, n: int, *params):
    """
    composite Gauss formula
    f: function to integrate
    xl : left limit
    xr : right limit
    N : number of steps
    n : number of nodes od small formulae
    *params: parameters of the variant — a, b, alpha, beta)
    """
    mesh = np.linspace(a, b, N + 1)
    return sum(quad_gauss(f, mesh[i], mesh[i + 1], n, *params) for i in range(N))


def equidist(n: int, xl: float, xr: float):
    if n == 1:
        return [0.5 * (xl + xr)]
    else:
        return np.linspace(xl, xr, n)


def runge(s1: float, s2: float, L: float, m: float):
    """ estimate m-degree error for s2 """
    error = np.abs((s2 - s1) / (L ** m - 1))
    return error


def aitken(s1: float, s2: float, s3: float, L: float):
    """
    estimate convergence order
    s1, s2, s3: consecutive composite quads
    return: convergence order estimation
    """
    m = -np.log((s3 - s2) / (s2 - s1)) / np.log(L)
    if str(m) == 'nan':
        return -1
    else:
        return m


def doubling_nc(f, xl: float, xr: float, n: int, tol: float, *params):
    """
    compute integral by doubling the steps number with theoretical convergence rate
    f : function to integrate
    xl : left limit
    xr : right limit
    n : nodes number in the small formula
    tol : error tolerance
    *params : arguments to pass to composite_quad function
    """
    # required local variables to return
    # S : computed value of the integral with required tolerance
    # N : number of steps for S
    # err : estimated error of S
    # iter_num : number of iterations (steps doubling)

    iter = 0
    N = 2
    err = np.inf

    while (err > tol):
        if iter == MAXITER:
            print("Convergence not reached!")
            return 0, 0, 10 * tol

        S1 = composite_quad(f, xl, xr, N, n, *params)
        S2 = composite_quad(f, xl, xr, N * 2, n, *params)
        err = runge(S1, S2, 2, n - 1)

        iter += 1
        N *= 2

    S = S2

    return N, S, err


def doubling_nc_aitken(f, xl: float, xr: float, n: int, tol: float, *params):
    """
    compute integral by doubling the steps number with Aitken estimation of the convergence rate
    f : function to integrate
    xl : left limit
    xr : right limit
    n : nodes number in the small formula
    tol : error tolerance
    *params : arguments to pass to composite_quad function
    """
    # required local variables to return
    # S : computed value of the integral with required tolerance
    # N : number of steps for S
    # err : estimated error of S
    # m : estimated convergence rate by Aitken for S
    # iter : number of iterations (steps doubling)

    iter = 0
    N = 2
    err = np.inf

    while (err > tol):
        if iter == MAXITER:
            print("Convergence not reached!")
            return 0, 0, 10 * tol

        S1 = composite_quad(f, xl, xr, N, n, *params)
        S2 = composite_quad(f, xl, xr, N * 2, n, *params)
        S3 = composite_quad(f, xl, xr, N * 4, n, *params)
        if aitken(S1, S2, S3, 2) > 0:
            m = aitken(S1, S2, S3, 2)
        else:
            m = n - 1
        err = runge(S1, S2, 2, m)

        iter += 1
        N *= 2

    S = S2

    return N, S, err, m


def doubling_gauss(f, xl: float, xr: float, n: int, tol: float, *params):
    """
    compute integral by doubling the steps number with theoretical convergence rate
    f : function to integrate
    xl : left limit
    xr : right limit
    n : nodes number in the small formula
    tol : error tolerance
    *params : arguments to pass to composite_quad function
    """
    # required local variables to return
    # S : computed value of the integral with required tolerance
    # N : number of steps for S
    # err : estimated error of S
    # iter : number of iterations (steps doubling)
    iter = 0
    N = 2
    err = np.inf

    while (err > tol):
        if iter == MAXITER:
            print("Convergence not reached!")
            return 0, 0, 10 * tol

        S1 = composite_gauss(f, xl, xr, N, n, *params)
        S2 = composite_gauss(f, xl, xr, N * 2, n, *params)
        err = runge(S1, S2, 2, n - 1)

        iter += 1
        N *= 2

    S = S2
    return N, S, err


def doubling_gauss_aitken(f, xl: float, xr: float, n: int, tol: float, *params):
    """
    compute integral by doubling the steps number with Aitken estimation of the convergence rate
    f : function to integrate
    xl : left limit
    xr : right limit
    n : nodes number in the small formula
    tol : error tolerance
    *params : arguments to pass to composite_quad function
    """
    # required local variables to return
    # S : computed value of the integral with required tolerance
    # N : number of steps for S
    # err : estimated error of S
    # m : estimated convergence rate by Aitken for S
    # iter : number of iterations (steps doubling)
    iter = 0
    N = 2
    err = np.inf

    while (err > tol):
        if iter == MAXITER:
            print("Convergence not reached!")
            return 0, 0, 10 * tol

        S1 = composite_gauss(f, xl, xr, N, n, *params)
        S2 = composite_gauss(f, xl, xr, N * 2, n, *params)
        S3 = composite_gauss(f, xl, xr, N * 4, n, *params)
        if aitken(S1, S2, S3, 2) > 0:
            m = aitken(S1, S2, S3, 2)
        else:
            m = n - 1
        err = runge(S2, S3, 2, m)

        iter += 1
        N *= 2

    S = S3
    return N, S, err, m


def optimal_nc(f, xl: float, xr: float, n: int, tol: float, *params):
    """ estimate the optimal step with Aitken and Runge procedures
    f : function to integrate
    xl : left limit
    xr : right limit
    n : nodes number in the small formula
    tol : error tolerance
    *params : arguments to pass to composite_quad function
    """
    # required local variables to return
    # S : computed value of the integral with required tolerance
    # N : number of steps for S
    # err : estimated error of S
    # iter : number of iterations (steps doubling)
    iter = 0
    N = 2
    err = np.inf

    S1 = composite_quad(f, xl, xr, N, n, *params)
    S2 = composite_quad(f, xl, xr, N*2, n, *params)
    h = (xr - xl)/N
    N_opt = h * ((tol * (1 - 2**(-n-1)))/np.abs(S2 - S1))**(1/( n+1 )) * 0.95
    Nh = math.ceil((xr - xl)/N_opt)
    print("Nh: ", Nh)
    while (err > tol):
        if iter == MAXITER:
            print("Convergence not reached!")
            return 0, 0, 10 * tol

        S1 = composite_quad(f, xl, xr, Nh, n, *params)
        S2 = composite_quad(f, xl, xr, Nh*2, n, *params)
        S3 = composite_quad(f, xl, xr, Nh*4, n, *params)
        if aitken(S1, S2, S3, 2) > 0:
            m = aitken(S1, S2, S3, 2)
        else:
            m = n + 1
        err = runge(S1, S2, 2, m)
        print("err: ", err)
        iter += 1
        Nh *= 2

    return Nh, S2, err
