import math

import numpy as np


def poly(weights):
    return lambda x: sum([i * (x**id) for id, i in enumerate(weights)])


# M(x) = c_1 + c_2 * x^1 + c_3 * x^2 + c_4 * x^3 + c_5 * sin(x * c_6 + c_7)
def sine_model(weights):
    c_1, c_2, c_3, c_4, c_5, c_6, c_7 = weights
    return (
        lambda x: c_1
        + c_2 * (x**1)
        + c_3 * (x**2)
        + c_4 * (x**3)
        + c_5 * np.sin(x * c_6 + c_7)
    )


def loss_funcs(X, Y, reg_part=lambda W: 0):
    funcs = []

    for i in range(len(X)):
        funcs.append(
            lambda W, i=i: (
                sum(
                    [
                        W[id] * j
                        for id, j in enumerate(
                            np.geomspace(1, X[i] ** (len(W) - 1), num=len(W))
                        )
                    ]
                )
                - Y[i]
            )
            ** 2
            + reg_part(W)
        )

    return funcs


def loss_funcs_sinus(X, Y, reg_part=lambda W: 0):
    funcs = []

    def _unpack(W, p):
        return W[p[0]] * p[1]

    def _loss_func_sinus(W, i):
        accum = 0
        for p in enumerate(np.geomspace(1, X[i] ** (len(W) - 1), num=len(W) - 3)):
            accum += _unpack(W, p)

        accum += W[-3] * np.sin((W[-2] * X[i] + W[-1]).astype(float))

        return accum - Y[i]

    def _gen(i):
        return lambda W: _loss_func_sinus(W, i)

    for i in range(len(X)):
        funcs.append(_gen(i))

    return funcs


def loss_func(funcs):
    return lambda W: sum([i(W) for i in funcs]) / len(funcs)
