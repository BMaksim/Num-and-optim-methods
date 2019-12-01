import numpy as np


def mngs(A, b, x0, eps):

    def func(x):
        return (1 / 2 * x.T @ A @ x + b.T @ x).item()

    x = np.array([x0])
    y = np.array([func(x0)])

    while True:
        q = A @ x0 + b
        m = -(q.T @ q)/(q.T @ A @ q)
        xk = x0 + m*q
        x = np.append(x, [xk], axis=0)
        y = np.append(y, func(xk))
        if abs(func(x0) - func(xk)) < eps:
            break
        else:
            x0 = xk
    return x, y


def mps(A, b, x0, eps):

    def func(x):
        return (1 / 2 * x.T @ A @ x + b.T @ x).item()

    x = np.array([x0])
    y = np.array([func(x0)])
    i = 0

    while True:
        E = np.eye(A.shape[0])
        e = E[i % A.shape[0]]
        e = np.reshape(e, (len(e), 1))
        m = -(e.T @ (A @ x0 + b))/(e.T @ A @ e)
        xk = x0 + m*e
        x = np.append(x, [xk], axis=0)
        y = np.append(y, func(xk))
        i += 1
        if abs(func(x0) - func(xk)) < eps:
            break
        else:
            x0 = xk
    return x, y
