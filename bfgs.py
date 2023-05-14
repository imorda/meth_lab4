import numpy as np
import typing as ty

import descent
from descent import wolfe_conditions, dichotomy

from funcs import f_bukin, f_matias


def rho_calc(y, s):
    return 1 / np.dot(y.T, s)


def bfgs(f: ty.Callable, df: ty.Callable, x_0, epochs, tol=1e-6, ls_tol=1e-6):
    i = 0
    E = np.eye(len(x_0))

    H_i = E  # reversed (^-1) hessian
    points = [x_0]
    x_i = x_0  # point
    df_x_i = df(f, x_0)  # derivative at x_i

    while i < epochs and np.linalg.norm(df_x_i) > tol:
        p_i = -np.dot(H_i, df_x_i)
        t, f_calls, df_calls = wolfe_conditions(f, df, x_i, p_i, tol=ls_tol)

        s = t * p_i
        x_i1 = x_i + s
        y = (df(f, x_i1) - df(f, x_i)).T

        rho = rho_calc(y, s)
        m1 = E - rho * np.dot(s, y.T)
        m2 = E - rho * np.dot(s.T, y)

        H_i = np.dot(m1, np.dot(H_i, m2)) + rho * np.dot(s, s.T)
        x_i = x_i1
        points.append(x_i)
        df_x_i = df(f, x_i)

        i += 1

    return points


# m --- is size of "batch" of precompiled data
def l_bfgs(f: ty.Callable, df: ty.Callable, x_0, epochs, m=5, tol=1e-9, ls_tol=1e-6):
    E = np.eye(len(x_0))

    H_i = E  # reversed (^-1) hessian
    points = [x_0]
    x_i = x_0  # point
    df_x_i = df(f, x_0)  # derivative at x_i

    ys = []
    ss = []

    for i in range(m):
        p_i = -np.dot(H_i, df_x_i)
        t, f_calls, df_calls = wolfe_conditions(f, df, x_i, p_i, tol=ls_tol)

        s = t * p_i
        x_i1 = x_i + s
        y = (df(f, x_i1) - df(f, x_i)).T

        ys.append(y)
        ss.append(s)

        rho = 1 / np.dot(y.T, s)
        m1 = E - rho * np.dot(s, y.T)
        m2 = E - rho * np.dot(s.T, y)

        H_i = np.dot(m1, np.dot(H_i, m2)) + rho * np.dot(s, s.T)
        x_i = x_i1
        points.append(x_i)
        df_x_i = df(f, x_i)

    i = m

    def alpha_calc(j, q):
        return rho_calc(ys[-j], ss[-j]) * np.dot(ss[-j].T, q)

    # calculating reversed hessian recursively
    while i < epochs and np.linalg.norm(df_x_i) > tol:
        q = df(f, x_i)

        for j in range(1, m):
            q = q - np.dot(alpha_calc(j, q), ys[-j])

        lambd = np.dot(ss[-1].T, ys[-1]) / np.dot(ys[-1].T, ys[-1])
        H_i = lambd * E
        p_i = np.dot(H_i, q)

        for j in range(m, 0, -1):
            betta = rho_calc(ys[-j], ss[-j]) * np.dot(ys[-j].T, p_i)
            p_i = p_i + np.dot(ss[-j], alpha_calc(j, q) - betta)

        p_i = -p_i

        t, f_calls, df_calls = wolfe_conditions(f, df, x_i, p_i, tol=1e-6)

        s = t * p_i
        x_i1 = x_i + s
        y = (df(f, x_i1) - df(f, x_i)).T

        ys.append(y)
        ss.append(s)

        x_i = x_i1
        points.append(x_i)
        df_x_i = df(f, x_i)

        i += 1

    return points
