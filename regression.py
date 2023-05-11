import numpy as np


def poly(weights):
    return lambda x: sum([i * (x ** id) for id, i in enumerate(weights)])

def loss_funcs(X, Y, reg_part = lambda W: 0):
  funcs = []

  for i in range(len(X)):
    funcs.append(lambda W, i=i: (sum([W[id] * j for id, j in enumerate(np.geomspace(1, X[i] ** (len(W) - 1), num=len(W)))]) - Y[i]) ** 2 + reg_part(W))

  return funcs

def loss_func(X, Y, reg_part = lambda W: 0):
  funcs = loss_funcs(X, Y, reg_part)
  return lambda W: sum([i(W) for i in funcs]) / len(funcs)
