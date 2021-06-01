#!/usr/bin/env python

import numpy as np

from contspy import Transient, plot_transient_results


class Diffusion(Transient):
    def __init__(self, L, N):
        Transient.__init__(self, L, N)
        self.k = 5.0e-02

    def Res(self, u):
        """
        The residual to solve for the system
        """
        return (u - self.u) / self.dt - self.k * np.dot(self.D2, u)

    def Jac(self, u):
        """
        The jacobian of the system
        """
        return np.diag(np.ones_like(u)) / self.dt - self.k * self.D2


def test_transient():
    problem = Diffusion(L=1.0, N=50)
    # Initial condition
    u0 = np.zeros(problem.N + 1)
    u0[10:40] = 1.0

    problem.run(u0, 0.1, 100, output_steps=True)


if __name__ == "__main__":
    test_transient()
    plot_transient_results("test_transient_out.csv")
