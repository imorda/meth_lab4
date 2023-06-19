import numpy

import descent
import bfgs
import visualization
import numpy as np
import dataset
import regression
import torch

from funcs import f_rosenbrock_torch, f_rosenbrock

t = visualization.visualize_multiple_descent_2args(
    {
        "bfgs": descent.scipy_descent(
            f_rosenbrock_torch,
            np.array([-10.0, -10.0]),
            method="bfgs",
            tol=1e-6,
            jac=True,
        )
    },
    f_rosenbrock,
)
print(t)
