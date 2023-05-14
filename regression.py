import numpy as np


def poly(weights):
    return lambda x: sum([i * (x ** id) for id, i in enumerate(weights)])

def loss_funcs(X, Y, reg_part = lambda W: 0):
  funcs = []

  for i in range(len(X)):
    funcs.append(lambda W, i=i: (sum([W[id] * j for id, j in enumerate(np.geomspace(1, X[i] ** (len(W) - 1), num=len(W)))]) - Y[i]) ** 2 + reg_part(W))

  return funcs

def loss_funcs_sinus(X, Y, reg_part = lambda W: 0):
  funcs = []

  def _unpack(W, p):
    return W[p[0]] * p[1]

  def _loss_func_sinus(W, i):
    accum = 0
    for p in enumerate(np.geomspace(1, X[i] ** (len(W) - 1), num=len(W) - 3)):
        accum += _unpack(W, p)

    accum += W[-3] * np.sin(W[-2] * X[-1] + W[-1])

    return (accum - Y[i]) ** 2 + reg_part(W)

  def _gen(i):
    return lambda W: _loss_func_sinus(W, i)

  for i in range(len(X)):
    funcs.append(_gen(i))

  return funcs

def loss_func(X, Y, reg_part = lambda W: 0):
  funcs = loss_funcs(X, Y, reg_part)
  return lambda W: sum([i(W) for i in funcs]) / len(funcs)
