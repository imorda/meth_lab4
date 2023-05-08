import numpy as np

import matplotlib.pyplot as plt

from linear_regression import poly

def_visualization_area = ((-12, 12), (-12, 12))  # Область отрисовки графиков ((start, stop) x dims)
def_visualization_resolution = 100  # Количество точек визуализации вдоль одной оси


def calc_axes(visualization_area, visualization_resolution):
    """
    Функция, создающая необходимое количество осей и равномерно заполняющая
    их точками для дальнейших расчётов по этим точкам для визуализации
    """
    axes = []
    for start, stop in visualization_area:
        axes.append(np.linspace(start, stop, visualization_resolution))
    return axes


def visualize_descent_2args(points,
                            f,
                            visualization_area=def_visualization_area,
                            visualization_resolution=def_visualization_resolution,
                            print_points=False):
    """
    Функция для визуализации работы градиентного спуска на функции f. Первым
    графиком выводится ломаная пар точек (iter, f(x)), получающихся в процессе работы метода,
    а на втором линии уровня функции f и точками отмечены шаги алгоритма.
    Параметры:
        points: шаги градиентного спуска
    """
    points = np.array(points)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    if print_points:
        print(points[:10])
        print("...")
        print(points[-10:])
    X, Y = np.meshgrid(*calc_axes(visualization_area, visualization_resolution))
    values = f(np.stack((X, Y)))
    ax2.contour(X, Y, values, levels=np.unique(
        np.sort(np.concatenate((f(points.T), np.linspace(np.amin(values), np.amax(values), 100))))))
    ax1.plot(f(points.T))
    ax1.set_yscale('symlog')
    ax1.grid()
    ax2.plot(points[:, 0], points[:, 1], '-')
    ax1.set_xlabel("# of iter", fontsize=20)
    ax1.set_ylabel("f(X)", fontsize=20)
    ax2.set_xlabel("X", fontsize=20)
    ax2.set_ylabel("Y", fontsize=20)
    fig.tight_layout(pad=5.0)


def visualize_descent(points,
                      f,
                      print_points=False):
    """
    Функция для визуализации работы градиентного спуска на функции f. Первым
    графиком выводится ломаная пар точек (iter, f(x)), получающихся в процессе работы метода,
    а также, опционально, печатаются координаты этих точек.
    Параметры:
        points: шаги градиентного спуска
    """
    points = np.array(points)
    fig, ax1 = plt.subplots()
    if print_points:
        print(points[:10])
        print("...")
        print(points[-10:])
    ax1.plot(f(points.T))
    ax1.set_yscale('symlog')
    ax1.grid()
    ax1.set_xlabel("# of iter", fontsize=20)
    ax1.set_ylabel("f(X)", fontsize=20)


def visualize_2arg(
        f,
        visualization_area=def_visualization_area,
        visualization_resolution=def_visualization_resolution):
    """
    Функция для отрисовки функции f 2х аргуметов
    """
    X, Y = np.meshgrid(*calc_axes(visualization_area, visualization_resolution))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, f(np.stack((X, Y))))
    ax.set_xlabel("$X$", fontsize=20)
    ax.set_ylabel("$Y$", fontsize=20)
    ax.set_zlabel("$f(x, y)$", fontsize=20, labelpad=-5)


def heatmap2d(x, y, arr: np.ndarray, nameX="", nameY=""):
    fig, ax = plt.subplots()
    plt.imshow(arr, cmap='viridis')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Number of iterations')
    ax.set_xticks(np.arange(0, x.shape[1]), labels=x[0])
    ax.set_yticks(np.arange(0, x.shape[0]), labels=y.T[0])
    ax.set_ylabel(nameY)
    ax.set_xlabel(nameX)


def print_stats(iters, func_calls=-1, grad_calls=-1, points=[(), ()]):
    print(f"""
    Начальная точка: {points[0]}
    Количество итераций: {iters}
    Количество вызовов функции: {func_calls}
    Количество вызовов градиента: {grad_calls}
    Конечная точка: {points[-1]}
  """)


def visualize_regression(weights: list, X, Y, x_name='', y_name='', line=False):
    p = poly(weights)

    x_axis = np.linspace(np.min(X), np.max(X), def_visualization_resolution)

    fig, ax = plt.subplots()

    if line:
        ax.plot(X, Y)
    else:
        ax.plot(X, Y, linestyle='none', marker='.')
    ax.set_ylabel(y_name)
    ax.set_xlabel(x_name)

    ax.plot(x_axis, p(x_axis))