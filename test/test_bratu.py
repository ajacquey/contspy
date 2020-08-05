#!/usr/bin/env python

import numpy as np

from contspy import Continuation, plot_continuation_results


class Bratu1d(Continuation):
    def Res(self, u, lmbda):
        """
      The residual to solve for the system
    """
        return np.dot(self.D2, u) + lmbda * np.exp(u)

    def dRes_dlmbda(self, u, lmbda):
        """
      The residual derivative wrt the lambda parameter
    """
        return np.exp(u)

    def Jac(self, u, lmbda):
        """
      The jacobian of the system
    """
        return self.D2 + lmbda * np.diag(np.exp(u))


def test_bratu():
    problem = Bratu1d(L=1.0, N=50)
    # Initial guess
    u0 = np.zeros(problem.N + 1)
    # Initial parameter value
    lmbda0 = 0.0

    problem.run(u0, lmbda0, 0.1, 500)


if __name__ == "__main__":
    test_bratu()
    plot_continuation_results("test_bratu_out.csv")
