import numpy as np


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