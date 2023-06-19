import numpy as np
import torch


def f_bukin(x):
    return 100 * np.sqrt(np.abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * np.abs(x[0] + 10)


def f_matias(x):
    return 0.28 * (x[0] ** 2 + x[1] ** 2) - 0.46 * x[0] * x[1]


def f_rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def f_rosenbrock_chunk():
    return [lambda x: 1 - x[0], lambda x: 10 * (x[1] - x[0] * x[0])]


def squared(funcs: list):
    return [(lambda x, i=i: i(x) ** 2) for i in funcs]


def torch_poly(a, x):
    return torch.matmul(
        torch.pow(torch.unsqueeze(x, -1), torch.arange(len(a), device=a.device)), a
    )


def torch_loss(x, y):
    return lambda w, x=x, y=y: torch.nn.functional.mse_loss(torch_poly(w, x), y)


def f_rosenbrock_torch(x):
    x = torch.tensor(x)
    x.requires_grad = True
    r = torch.sum(100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2
                  + (1 - x[..., :-1]) ** 2)
    r.backward()
    return r.detach().numpy(), x.grad.detach().numpy()


# def random_polynom(x):
#     return 5*x[0] +
