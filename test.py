import descent
import bfgs
import visualization
from funcs import f_rosenbrock, f_rosenbrock_chunk
import numpy as np

if __name__ == '__main__':
    visualization.visualize_multiple_descent_2args_wh_time(
        {
            'bfgs': lambda: descent.scipy_descent(f_rosenbrock, np.array([-10.0, -10.0]), method='bfgs', tol=1e-6,
                                                  jac='2-point'),
        },
        f_rosenbrock,
    )
