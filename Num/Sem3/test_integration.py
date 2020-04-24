import numpy as np
import matplotlib.pyplot as plt


from integration import (quad, moments, quad_gauss, composite_quad, composite_gauss, 
                           equidist, runge, aitken, optimal_nc, doubling_nc,
                           doubling_nc_aitken, doubling_gauss, doubling_gauss_aitken)


def call_counter(func):
    def helper(*args, **kwargs):
        helper.calls += 1
        return func(*args, **kwargs)
    helper.calls = 0
    return helper


def get_log_error(x, y, axis=1):
    x = np.array(x)
    y = np.array(y)
    if len(x.shape) == 1:
        return np.log10(np.abs(x - y))
    else:
        return np.log10(np.linalg.norm(x - y))




variant = 6
MAXITER = 12
# Содержание
#
# 1. Проверка АСТ ИКФ Ньютона — Котса с числом узлов от 1 до 6 (1 узел берётся в середине):
# 1.a. Применение для формулы с единичным весом;
# 1.b. Применение для формулы с конкретным весом, стемящимся к бесконечности в a или в b.
#
# 2. Проверка АСТ формулы типа Гаусса с числом узлов от 1 до 3:
# 2.a. Применение для формулы с единичным весом;
# 2.b. Применение для формулы с конкретным весом, стемящимся к бесконечности в a или в b.
#
# 3. Проверка сходимости СКФ Ньютона — Котса с тремя узлами половинным делением:
# 3.a. Использование теоретической скорости сходимости в правиле Рунге;
# 3.b. Использование скорости сходимости, вычисленной по правилу Эйткена;
# 3.c. Проверка стремления скорости сходимости к теоретической (или нет).
#
# 4. Проверка сходимости СКФ Гаусса с тремя узлами половинным делением:
# 4.a. Использование теоретической скорости сходимости в правиле Рунге;
# 4.b. Использование скорости сходимости, вычисленной по правилу Эйткена;
# 4.c. Проверка стремления скорости сходимости к теоретической.
#
# 5. Использование оптимального шага в СКФ Ньютона — Котса:
# 5.a. Проверка того, что предложенный оптимальный шаг даёт нужную точность;
# 5.b. Проверка того, что предложенный оптимальный шаг не слишком мал.

# Реализация

# 1. Проверка АСТ ИКФ Ньютона — Котса с числом узлов от 1 до 6 (1 узел берётся в середине)
def test_nc_degree():
    """Newton-Cotes algebraic degree of accuracy check"""

    # 1.a. Применение для формулы с единичным весом
    a, b, alpha, beta, f = params(variant)
    alpha, beta = 0, 0

    max_s = 7  # проверяем для одночленов степени до 6 включительно

    for s in range(max_s):
        f = lambda x: x ** s  # одночлен x^s
        J = (b ** (s + 1) - a ** (s + 1)) / (s + 1)  # точное значение интеграла
        n_range = range(1, 7)  # число узлов от 1 до 6

        S = [quad(f, a, b, equidist(n, a, b)) for n in n_range]  # применяем ИКФ на равномерной сетке
        # S = [quad(f, a, b, a + (b - a) * np.random.random(n)) for n in n_range]
        accuracy = get_log_error(S, J * np.ones_like(S))  # считаем точную погрешность
        accuracy[np.isinf(accuracy)] = -17  # если погрешность 0, то выводим точность в 17 знаков

        # check accuracy is good enough
        for n, acc in zip(n_range, accuracy):
            if n >= s + (s + 1) % 2:  # с нечётным числом n узлов АСТ должно быть равно n, иначе n-1
                assert acc < -6

        plt.plot(n_range, accuracy, '.:', label=f'x^{s}')

    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('node_count')
    plt.suptitle(f'Accuracy test for constant weight')
    plt.show()

    # 1.b. Применение для формулы с конкретным весом, стемящимся к бесконечности в a или в b
    a, b, alpha, beta, f = params(variant)

    max_s = 7  # проверяем для одночленов степени до 6 включительно

    J = moments(max_s - 1, a, b, a, b, alpha, beta)  # точные значение интегралов
    for s in range(max_s):
        f = lambda x: x ** s  # одночлен x^s
        n_range = range(1, 7)  # число узлов от 1 до 6

        S = [quad(f, a, b, equidist(n, a, b), a, b, alpha, beta) for n in n_range]  # применяем ИКФ на равномерной сетке
        accuracy = get_log_error(S, J[s])  # считаем точную погрешность
        accuracy[np.isinf(accuracy)] = -17  # если погрешность 0, то выводим точность в 17 знаков

        # check accuracy is good enough
        for n, acc in zip(n_range, accuracy):  # АСТ должно быть равно n-1
            if n >= s + 1:
                assert acc < -6

        plt.plot(n_range, accuracy, '.:', label=f'x^{s}')

    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('node_count')
    plt.suptitle(f'Accurace test for non-symmetric weight')
    plt.show()


# 2. Проверка АСТ формулы типа Гаусса с числом узлов от 1 до 3
def test_gauss_degree():
    """Gauss algebraic degree of accuracy check"""

    # 2.a. Применение для формулы с единичным весом
    a, b, alpha, beta, f = params(variant)
    alpha, beta = 0, 0

    max_s = 7  # проверяем для одночленов степени до 6 включительно

    for s in range(max_s):
        f = lambda x: x ** s  # одночлен x^s
        J = (b ** (s + 1) - a ** (s + 1)) / (s + 1)  # точное значение интеграла
        n_range = range(1, 4)  # число узлов от 1 до 3

        S = [quad_gauss(f, a, b, n) for n in n_range]  # применяем ИКФ на равномерной сетке
        accuracy = get_log_error(S, J * np.ones_like(S))  # считаем точную погрешность
        accuracy[np.isinf(accuracy)] = -17  # если погрешность 0, то выводим точность в 17 знаков

        # check accuracy is good enough
        for n, acc in zip(n_range, accuracy):
            if s <= 2 * n - 1:  # АСТ должно быть равно 2n-1
                assert acc < -6

        plt.plot(n_range, accuracy, '.:', label=f'x^{s}')

    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('node_count')
    plt.suptitle(f'Accuracy test for constant weight')
    plt.show()

    # 2.b. Применение для формулы с конкретным весом, стемящимся к бесконечности в a или в b
    a, b, alpha, beta, f = params(variant)

    max_s = 7  # проверяем для одночленов степени до 6 включительно

    J = moments(max_s - 1, a, b, a, b, alpha, beta)  # точные значение интегралов
    for s in range(max_s):
        f = lambda x: x ** s  # одночлен x^s
        n_range = range(1, 4)  # число узлов от 1 до 3

        S = [quad_gauss(f, a, b, n, a, b, alpha, beta) for n in n_range]  # применяем ИКФ на равномерной сетке
        accuracy = get_log_error(S, J[s])  # считаем точную погрешность
        accuracy[np.isinf(accuracy)] = -17  # если погрешность 0, то выводим точность в 17 знаков

        # check accuracy is good enough
        for n, acc in zip(n_range, accuracy):  # АСТ должно быть равно n-1
            if s <= 2 * n - 1:  # АСТ должно быть равно 2n-1
                assert acc < -6

        plt.plot(n_range, accuracy, '.:', label=f'x^{s}')

    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('node_count')
    plt.suptitle(f'Accurace test for non-symmetric weight')
    plt.show()


# 3. Проверка сходимости СКФ Ньютона — Котса с тремя узлами половинным делением
def test_nc_convergence():
    n = 3  # выбираем трёхточечную формулу
    a, b, alpha, beta, f = params(variant)
    tol = 1e-8  # требование на точность

    # 3.a. Использование теоретической скорости сходимости в правиле Рунге
    print("\nConvergence with step doubling and m = 3 in Runge rule")

    N, S, err = doubling_nc(f, a, b, n, tol, a, b, alpha, beta)

    print("Computed value: ", S)
    assert N > 0
    print("Steps made: ", N)
    print("Exact error: ", np.abs(S - exactint(variant)))
    assert np.abs(S - exactint(variant)) < tol


    # 3.b. Использование скорости сходимости, вычисленной по правилу Эйткена
    print("\nConvergence with step doubling and Aitken convergence estimation for Runge rule")

    N, S, err, m = doubling_nc_aitken(f, a, b, n, tol, a, b, alpha, beta)

    print("Computed value: ", S)
    assert N > 0
    print("Steps made: ", N)
    print("Exact error: ", np.abs(S - exactint(variant)))
    assert np.abs(S - exactint(variant)) < tol

    # 3.c. Проверка стремления скорости сходимости к теоретической (или выше)
    print("Final convergence order: ", m)
    assert m - 3 > -0.1 # m может быть больше теоретических 3, но не должна быть меньше (0.1 — это допуск)
    # здесь получается, что скорость между 3 и 4 — это вполне объяснимо (подумайте)

# 4. Проверка сходимости СКФ Гаусса с тремя узлами половинным делением
def test_gauss_convergence():
    n = 3  # выбираем трёхточечную формулу
    a, b, alpha, beta, f = params(variant)
    tol = 1e-8  # требование на точность

    # 4.a. Использование теоретической скорости сходимости в правиле Рунге
    print("\nConvergence with step doubling and m = 6 in Runge rule")

    N, S, err = doubling_gauss(f, a, b, n, tol, a, b, alpha, beta)

    print("Computed value: ", S)
    assert N > 0
    print("Steps made: ", N)
    print("Estimated error: ", err)
    print("Exact error: ", np.abs(S - exactint(variant)))
    # assert np.abs(S2 - exactint(variant)) < tol # Может быть неверно!
    if np.abs(S - exactint(variant)) > tol:
        print("ТОЧНОСТЬ НЕ ДОСТИГНУТА! СМ. СЛЕДУЮЩИЙ ТЕСТ С ОЦЕНКОЙ ЭЙТКЕНА!")

    # 4.b. Использование скорости сходимости, вычисленной по правилу Эйткена
    print("\nConvergence with step doubling and Aitken convergence estimation for Runge rule")

    N, S, err, m = doubling_gauss_aitken(f, a, b, n, tol, a, b, alpha, beta)

    print("Computed value: ", S)
    assert N > 0
    print("Steps made: ", N)
    print("Estimated error: ", err)
    print("Exact error: ", np.abs(S - exactint(variant)))
    assert np.abs(S - exactint(variant)) < tol

    # 4.c. Проверка стремления скорости сходимости к теоретической
    print("Final convergence order: ", m)
    assert m - 6 > -0.1  # m должна стремиться к теоретическим 6 (0.1 — это допуск)
    # здесь часто получается, что скорость выше, чем 6. Это странно, но вот.


# 5. Использование оптимального шага в СКФ Ньютона — Котса:
def test_optimal():
    n = 3  # выбираем трёхточечную формулу
    a, b, alpha, beta, f = params(variant)
    tol = 1e-8  # требование на точность

    # 5.a. Проверка того, что предложенный оптимальный шаг даёт нужную точность
    print("\nOptimal step prediction with Aitken convergence estimation for Runge rule")

    N, S, err = optimal_nc(f, a, b, n, tol, a, b, alpha, beta)

    print("Computed value: ", S)
    print("Steps made: ", N)
    print("Estimated error: ", err)
    print("Exact error: ", np.abs(S - exactint(variant)))
    assert np.abs(S - exactint(variant)) < tol

    # 5.b. Проверка того, что предложенный оптимальный шаг не слишком мал
    N2, S2, err2, m2 = doubling_nc_aitken(f, a, b, n, tol, a, b, alpha, beta)
    assert N < 2 * N2
    # Вообще должно быть N < N2, но бывает так, что мы удваивали число оптимальных,
    # если оптимальный был не совсем хорош, вместо предложения оптимального числа немного большего текущего.
    # Кроме того из-за гарантийного множителя может получиться оптимальное число чуть больше, чем действительно нужное,
    # которое ровно попало в половинном делении
#end test_optimal


def params(v):
    """ Chooses parameters according to the variant number"""
    KC = [2.0, 3.0, 2.5, 3.0, 1.0, 4.0, 4.5, 3.7, 3.0, 1.3, 0.5, 4.0, 2.0, 3.0, 3.5, 2.7, 6.0, 4.0, 0.5, 1.5, 3.0, 5.0,
          2.5, 5.7]
    FC = [2.5, 0.5, 2.0, 3.5, 1.5, 0.5, 7.0, 1.5, 1.5, 3.5, 2.0, 2.5, 3.5, 2.5, 0.7, 3.5, 1.5, 2.5, 3.0, 3.7, 2.5, 0.3,
          5.7, 2.5]
    EC = [1 / 3, 1 / 4, 2 / 3, 4 / 3, 2 / 3, -5 / 4, -2 / 3, -4 / 3, 1 / 4, 2 / 3, 2 / 5, 4 / 7, 5 / 3, 7 / 4, -5 / 3,
          -7 / 3, 5 / 3, 5 / 4, 2 / 5, 4 / 7, 4 / 3, -7 / 4, -4 / 3, -4 / 7]
    KS = [4.0, 5.0, 4.0, 2.0, 3.0, 2.0, 1.4, 2.4, 4.0, 6.0, 2.4, 2.5, 3.0, 5.0, 2.4, 4.4, 2.0, 2.5, 4.0, 3.0, 4.0, 7.0,
          2.4, 4.4]
    FS = [3.5, 2.5, 3.5, 3.5, 5.5, 4.5, 1.5, 4.5, 3.5, 4.5, 1.5, 5.5, 1.5, 0.5, 5.5, 2.5, 0.5, 1.5, 3.5, 2.5, 5.5, 0.5,
          2.5, 4.3]
    ES = [-3, -1 / 3, -3.0, -2 / 3, -2.0, 1 / 8, -1 / 3, 2 / 3, -3.0, -1 / 8, -6.0, -0.6, -4.0, 3 / 8, -3 / 4, 5 / 3,
          -1.3, -2 / 7, -3.0, 3 / 4, -3.5, 2 / 3, -3 / 3, 2 / 7]
    KL = [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 4.3, 0.0, 0.0, 0.0, 0.0, 5.4, 5.0, 3.0, 3.0, 0.0, 0.0,
          0.0, 0.0]
    K0 = [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0,
          7.0, 5.0]
    A = [1.5, 1.7, 0.1, 1.0, 2.5, 1.3, 2.1, 1.8, 2.5, 0.7, 1.1, 1.8, 1.5, 2.3, 1.1, 2.8, 3.5, 2.7, 1.1, 1.5, 2.5, 0.5,
         0.2, 0.8]
    B = [3.3, 3.2, 2.3, 3.0, 4.3, 2.2, 3.3, 2.3, 3.3, 3.2, 2.5, 2.9, 2.3, 2.9, 2.3, 4.3, 3.7, 3.2, 2.3, 3.0, 4.3, 2.2,
         3.1, 1.3]
    Al = [1 / 3, 0, 1 / 5, 0, 2 / 7, 0, 2 / 5, 0, 2 / 3, 0, 2 / 5, 0, 1 / 5, 0, 4 / 5, 0, 2 / 3, 0, 2 / 5, 0, 2 / 7, 0,
          3 / 5, 0]
    Be = [0, 1 / 4, 0, 1 / 6, 0, 5 / 6, 0, 3 / 5, 0, 1 / 4, 0, 4 / 7, 0, 2 / 5, 0, 3 / 7, 0, 3 / 4, 0, 5 / 6, 0, 3 / 5,
          0, 4 / 7]
    if v < 1 or v > 24:
        return 0
    v = v - 1

    def f(x):
        return KC[v] * np.cos(FC[v] * x) * np.exp(EC[v] * x) + KS[v] * np.sin(FS[v] * x) * np.exp(ES[v] * x) \
               + KL[v] * x + K0[v]

    return A[v], B[v], Al[v], Be[v], f


def exactint(v):
    """ Gives an "exact" value for the variant number"""
    exactvals = [7.077031437995793610263911711602477164432,
                 11.83933565874812191864851199716726555747,
                 3.578861536040539915439859609644293194417,
                 -41.88816344003630606891235682900290027460,
                 10.65722906811476196545133157861241468330,
                 10.83954510946909397740794566485262705081,
                 4.461512705331194112840828080521604042844,
                 1.185141974956241824914878594317090726677,
                 20.73027110955223102601793414048307154080,
                 24.14209267859915860831257727834195698139,
                 18.60294785731848208626949366919856494853,
                 57.48462064655285571820619434508191055583,
                 32.21951452884234295708696008290380201405,
                 348.8181344253911360363589124705960119778,
                 27.56649553650691538577624747358600185818,
                 -3246.875926327328894367882485073567528036,
                 2308.287524452809436132810373422088766896,
                 78.38144689028315358349839381435476300192,
                 8.565534222407634006755741863827588778916,
                 161.7842904748235945321114040034846373768,
                 -262.7627605704703725392313581988618564726,
                 69.34894027882668315183332391303280381054,
                 28.98579534502018413362688379858804448110,
                 -4.249393101145035941757249850984813018073]
    if v < 1 or v > 24:
        return 0
    return exactvals[v-1]

test_optimal()