from vector import Vector


# objective function
def f(point):
    x, y = point
    return x ** 2 + x * y + y ** 2 - 6 * x - 9 * y


def nelder_mead(alpha=1, beta=0.5, gamma=2, maxiter=10):
    # initialization
    v1 = Vector(0, 0)
    v2 = Vector(1.0, 0)
    v3 = Vector(0, 1)

    for i in range(maxiter):
        adict = {v1: f(v1.c()), v2: f(v2.c()), v3: f(v3.c())}
        points = sorted(adict.items(), key=lambda x: x[1])

        b = points[0][0]
        g = points[1][0]
        w = points[2][0]

        mid = (g + b) / 2

        # reflection
        xr = mid + alpha * (mid - w)
        if f(xr.c()) < f(g.c()):
            w = xr
        else:
            if f(xr.c()) < f(w.c()):
                w = xr
            c = (w + mid) / 2
            if f(c.c()) < f(w.c()):
                w = c
        if f(xr.c()) < f(b.c()):

            # expansion
            xe = mid + gamma * (xr - mid)
            if f(xe.c()) < f(xr.c()):
                w = xe
            else:
                w = xr
        if f(xr.c()) > f(g.c()):

            # contraction
            xc = mid + beta * (w - mid)
            if f(xc.c()) < f(w.c()):
                w = xc

        # update points
        v1 = w
        v2 = g
        v3 = b
    return b


print("Result of Nelder-Mead algorithm: ")
xk = nelder_mead()
print("Best poits is: %s" % (xk))