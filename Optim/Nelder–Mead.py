import numpy as np


def ros(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2


alpha, beta, gamma, var = 1, 0.5, 2, 0.001
p1, p2, p3 = np.array([1, 0]), np.array([0, 0]), np.array([0, 1])

values = {'p1': ros(p1), 'p2': ros(p2), 'p3': ros(p3)}
values = sorted(values.items(), key=lambda x: x[1])


print(values)

