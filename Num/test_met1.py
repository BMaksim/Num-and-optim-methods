import pytest
import numpy as np
import matplotlib.pyplot as plt
import met1

default_student = 3


@pytest.mark.parametrize("student", [default_student])  # student num
@pytest.mark.parametrize("n_dim", [2, 3])               # dimension number
def test_met1(student, n_dim):
    print(f'running met1 test for student #{student}')
    N = student

    A = np.array([[4,  1,      1],
                  [1,  6+.2*N, -1],
                  [1,  -1,      8+.2*N]],
                 dtype='float')[:n_dim, :n_dim]
    b = np.array([1, -2, 3], dtype='float').reshape(-1, 1)[:n_dim, :]
    x0 = np.array([0, 0, 0], dtype='float').reshape(-1, 1)[:n_dim, :]

    eps_y = 1e-6
    eps_x = 1e-3

    methods = ['mngs', 'mps']
    styles = ['mo-', 'b.:']
    plt.figure()
    plt.title(f'Результаты для размерности {n_dim}')
    plt.xlabel('номер итерации')
    plt.ylabel('точность')
    for i, method in enumerate(methods):
        X, Y = getattr(met1, method)(A, b, x0, eps_y)

        x1 = np.linalg.solve(A, -b)
        y1 = (1/2 * x1.T @ A @ x1 + b.T @ x1).item()

        assert np.equal(x0, X[0]).all(), 'X should start with initial point'
        assert np.linalg.norm(x1 - X[-1]) < eps_x, 'last X should be close enough to the optimum'
        assert np.linalg.norm(y1 - Y[-1]) < eps_y, 'last Y should be close enough to the optimum'

        plt.plot(-np.log10([y - y1 for y in Y]), styles[i])
    plt.legend(methods)
    plt.show()
