import numpy as np


def cheb(N, L=1.0):
    """
    Chebyshev polynomial differentiation matrix.
    Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
    INPUT
    N: discretization points number (must be even)
    L: size of the domain (default is 1)
    OUTPUT
    D: Differentiation matrix
    D2: Differentiation matrix second order
    x: Spatial discretization
  """
    assert N % 2 == 0, "N should be even!"  # only when N is even!
    x = L / 2.0 * np.cos(np.pi * np.arange(0, N + 1) / N)
    x[int(N / 2)] = 0.0
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0
    c = c * (-1.0) ** np.arange(0, N + 1)
    c = c.reshape(N + 1, 1)
    X = np.tile(x.reshape(N + 1, 1), (1, N + 1))
    dX = X - X.T
    D = np.dot(c, 1.0 / c.T) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    D2 = np.dot(D, D)
    return D, D2, x


def applyBC(array):
    """
    Apply boundary conditions
    INPUT:
    D: Differentiation matrix
    D2: Differentiation matrix second order
    x: Spatial discretization
    OUTPUT:
    D: Differentiation matrix
    D2: Differentiation matrix second order
    x: Spatial discretization
  """
    arr_shape = array.shape
    if len(arr_shape) == 1:
        N = len(array) - 1
        return array[1:N]
    elif len(arr_shape) == 2:
        N = len(array[0, :]) - 1
        return array[1:N, 1:N]
    else:
        raise Exception("Cannot apply BC, invalid ipunt array!")
