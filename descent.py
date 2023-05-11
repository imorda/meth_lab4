import math
import random
import typing as ty
import numpy as np

from profiler import Num, to_num


def numeric_gradient(f, x, h=1e-6):
    """
    Функция для численного вычисления градиента функции f в точке x.
    Параметры:
        x: точка, в которой нужно вычислить градиент
        h: малое число для вычисления приближенного значения производной
    """
    n = x.shape[0]  # число переменных
    grad = np.zeros(n)  # инициализация градиента
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += h  # прибавляем малое число к i-ой координате
        x_minus = x.copy()
        x_minus[i] -= h  # вычитаем малое число из i-ой координаты
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)  # вычисляем приближенное значение производной
    return grad


def constant_lr_decay(lr):
    def f(*_, **__):
        return lr

    return f


def exp_decay(lr0, k):
    def f(epoch):
        return lr0 * math.exp(-k * epoch)

    return f


def step_decay(lr0, drop, epochs_drop):
    def f(epoch):
        return lr0 * (drop ** (epoch // epochs_drop))

    return f

def adam_minibatch_descent(f: ty.List[ty.Callable], df, x0, decay, tol,
                           n_epochs,
                           batch_size=1, betta1=0.9, betta2=0.9, silent=False, global_stats=False):
    random.shuffle(f)
    n = len(f)
    points = [x0]
    grad_square_steps = [np.array([Num(0)] * len(x0), dtype=Num)]
    grad_steps = [np.array([Num(0)] * len(x0), dtype=Num)]
    eps = 1e-8

    last_grads = np.array([Num(0.0) for _ in range(len(x0))], dtype=Num)
    num_grads = 0
    last_epoch = 0

    for i in range(n // batch_size * n_epochs):
        if i * batch_size // n > last_epoch:
            if np.linalg.norm(last_grads / num_grads) < tol:
                if not silent:
                    print(np.linalg.norm(last_grads / num_grads))
                break
            last_grads = 0.0
            num_grads = 0
            last_epoch = i * batch_size // n

        x = points[-1]
        part_grad = np.array(to_num(sum(df(f[(i * batch_size + j) % n], x) for j in range(batch_size)) / batch_size),
                             dtype=Num)

        last_grads += part_grad
        num_grads += 1

        grad_square_steps.append(grad_square_steps[-1] * betta1 + part_grad ** 2 * (1 - betta1))
        grad_steps.append(grad_steps[-1] * betta2 + part_grad * (1 - betta2))
        grad_step_normalized = grad_steps[-1] / (1 - betta1 ** (i + 1))
        grad_square_step_normalized = grad_square_steps[-1] / (1 - betta2 ** (i + 1))
        points.append(
            x - decay(i * batch_size // n) * grad_step_normalized * (grad_square_step_normalized + eps) ** (-1 / 2))

    if global_stats:
        global _stats
        _stats.append(len(points))
    return points

# TODO Реализация метода Gauss-Newton