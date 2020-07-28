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
  assert (N%2 == 0), "N should be even!" # only when N is even!
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

def applyBC(D, D2, x):
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
  N = len(x) - 1
  return D[1:N,1:N], D2[1:N,1:N], x[1:N]