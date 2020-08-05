#!/usr/bin/env python

import numpy as np

from contspy import Continuation, plot_continuation_results


class Coupled(Continuation):
    def __init__(self, L, N, nvar):
        Continuation.__init__(self, L, N, nvar)
        self.Ar = 5.0

    def Res(self, u, lmbda):
        """
      The residual to solve for the system
    """
        res0 = self.Res0(u, lmbda)
        res1 = self.Res1(u, lmbda)

        return np.hstack((res0, res1))

    def Res0(self, u, lmbda):
        """
        Same residual as in test_S_coupled.py
    """
        u0, _ = np.split(u, 2)
        return np.dot(self.D2, u0) + lmbda * np.exp(np.divide(self.Ar * u0, 1.0 + u0))

    def Res1(self, u, lmbda):
        """
        Only diffusion here
    """
        _, u1 = np.split(u, 2)
        return np.dot(self.D2, u1)

    def Jac(self, u, lmbda):
        """
      The jacobian of the system
    """
        jac00 = self.Jac00(u, lmbda)
        jac01 = self.Jac01(u, lmbda)
        jac10 = self.Jac10(u, lmbda)
        jac11 = self.Jac11(u, lmbda)
        return np.block([[jac00, jac01], [jac10, jac11]])

    def Jac00(self, u, lmbda):
        """
        Derivative of Res0 wrt u0
    """
        u0, _ = np.split(u, 2)
        return self.D2 + lmbda * self.Ar * np.diag(
            np.divide(np.exp(np.divide(self.Ar * u0, 1.0 + u0)), np.square(1.0 + u0))
        )

    def Jac01(self, u, lmbda):
        """
        Derivative of Res0 wrt u1
    """
        u0, u1 = np.split(u, 2)
        return np.zeros((len(u0), len(u1)))

    def Jac10(self, u, lmbda):
        """
        Derivative of Res1 wrt u0
    """
        u0, u1 = np.split(u, 2)
        return np.zeros((len(u1), len(u0)))

    def Jac11(self, u, lmbda):
        """
        Derivative of Res1 wrt u1
    """
        return self.D2


def test_coupled():
    problem = Coupled(L=1.0, N=50, nvar=2)
    # Initial guess
    u0 = np.zeros(problem.N + 1)
    u1 = np.zeros(problem.N + 1)
    u = np.hstack((u0, u1))
    # Initial parameter value
    lmbda0 = 0.0

    problem.run(u, lmbda0, 0.1, 800)


if __name__ == "__main__":
    test_coupled()
    plot_continuation_results("test_coupled_out.csv", nvar=2)
