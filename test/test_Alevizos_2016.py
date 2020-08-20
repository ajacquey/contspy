#!/usr/bin/env python

import numpy as np

from contspy import Continuation, plot_continuation_results


class Alevizos(Continuation):
    def __init__(self, L, N, Le):
        Continuation.__init__(self, L, N, 2)
        self.Arm = 10.0
        self.Arc = 20.0
        self.alpha = 4.0
        self.Da = 1.0e-04
        self.mur = 3.0e-03
        self.phi = 0.1
        self.Le = Le

    def Res(self, u, lmbda):
        """
      The residual to solve for the system
    """
        res0 = self.Res0(u, lmbda)
        res1 = self.Res1(u, lmbda)

        return np.hstack((res0, res1))

    def Res0(self, u, lmbda):
        """
        Temperature equation as in Alevizos et al. (2016) eq. 6
    """
        u0, u1 = np.split(u, 2)
        return (
            np.dot(self.D2, u0)
            + lmbda * np.exp(np.divide(self.Arm * u0, 1.0 + u0) - self.alpha * u1)
            - self.Da * (1.0 - self.phi) * np.exp(np.divide(self.Arc * u0, 1.0 + u0))
        )

    def Res1(self, u, lmbda):
        """
        Only diffusion here
    """
        u0, u1 = np.split(u, 2)
        return np.dot(self.D2, u1) / self.Le + self.mur * (1.0 - self.phi) * np.exp(
            np.divide(self.Arc * u0, 1.0 + u0)
        )

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
        u0, u1 = np.split(u, 2)
        return (
            self.D2
            + lmbda
            * self.Arm
            * np.diag(
                np.divide(
                    np.exp(np.divide(self.Arm * u0, 1.0 + u0) - self.alpha * u1),
                    np.square(1.0 + u0),
                )
            )
            - self.Da
            * (1.0 - self.phi)
            * self.Arc
            * np.diag(
                np.divide(
                    np.exp(np.divide(self.Arm * u0, 1.0 + u0) - self.alpha * u1),
                    np.square(1.0 + u0),
                )
            )
        )

    def Jac01(self, u, lmbda):
        """
        Derivative of Res0 wrt u1
    """
        u0, u1 = np.split(u, 2)
        return (
            -lmbda
            * self.alpha
            * np.diag(np.exp(np.divide(self.Arm * u0, 1.0 + u0) - self.alpha * u1))
        )

    def Jac10(self, u, lmbda):
        """
        Derivative of Res1 wrt u0
    """
        u0, u1 = np.split(u, 2)
        return (
            self.mur
            * (1.0 - self.phi)
            * self.Arc
            * np.diag(
                np.divide(
                    np.exp(np.divide(self.Arc * u0, 1.0 + u0)), np.square(1.0 + u0)
                )
            )
        )

    def Jac11(self, u, lmbda):
        """
        Derivative of Res1 wrt u1
    """
        return self.D2 / self.Le


def test_alevizos():
    problem = Alevizos(L=2.0, N=40, Le=0.8)
    # Initial guess
    u0 = np.zeros(problem.N + 1)
    u1 = np.zeros(problem.N + 1)
    u = np.hstack((u0, u1))
    # Initial parameter value
    lmbda0 = 0.0

    problem.run(u, lmbda0, 0.02, 500)


if __name__ == "__main__":
    test_alevizos()
    plot_continuation_results("test_Alevizos_2016_out.csv", nvar=2)
