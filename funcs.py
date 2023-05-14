import numpy as np


def f_bukin(x):
    return 100 * np.sqrt(np.abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * np.abs(x[0] + 10)


def f_matias(x):
    return 0.28 * (x[0] ** 2 + x[1] ** 2) - 0.46 * x[0] * x[1]
