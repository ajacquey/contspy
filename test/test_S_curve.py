#!/usr/bin/env python

import numpy as np

from contspy import Continuation, plot_continuation_results


class SCurve(Continuation):
    def __init__(self, L, N):
        Continuation.__init__(self, L, N)
        self.Ar = 5.0

    def Res(self, u, lmbda):
        """
      The residual to solve for the system
    """
        return np.dot(self.D2, u) + lmbda * np.exp(np.divide(self.Ar * u, 1.0 + u))

    def Jac(self, u, lmbda):
        """
      The jacobian of the system
    """
        return self.D2 + lmbda * self.Ar * np.diag(
            np.divide(np.exp(np.divide(self.Ar * u, 1.0 + u)), np.square(1.0 + u))
        )


def test_s_curve():
    problem = SCurve(L=1.0, N=50)
    # Initial guess
    u0 = np.zeros(problem.N + 1)
    # Initial parameter value
    lmbda0 = 0.0

    problem.run(u0, lmbda0, 0.1, 800)


if __name__ == "__main__":
    test_s_curve()
    plot_continuation_results("test_S_curve_out.csv")
