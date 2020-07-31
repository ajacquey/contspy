#!/usr/bin/env python

import sys

# import matplotlib.pyplot as plt
import numpy as np

from contspy import main

sys.path.insert(0, "/Users/ajacquey/projects/contspy/")


class Bratu1d(main.Continuation):
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

    # # Convert lists to numpy arrays
    # lmbda = np.array(lmbda_list)
    # val_norm = np.array(values_list)
    # stability = np.array(stability_list)
    # val_norm_stable = np.NaN * np.ones_like(val_norm)
    # val_norm_unstable = np.NaN * np.ones_like(val_norm)
    # val_norm_stable[stability] = val_norm[stability]
    # val_norm_unstable[~stability] = val_norm[~stability]

    # # Plot
    # plt.rc("text", usetex=True)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.set_xlabel(r"$\lambda$")
    # ax1.set_ylabel(r"$||u||_{\infty}$")
    # ax1.grid()

    # ax1.plot(lmbda, val_norm_stable, "-", color="blue")
    # ax1.plot(lmbda, val_norm_unstable, "--", color="blue")
    # ax1.plot(lmbda[saddle_list], val_norm[saddle_list], "o", color="blue")
    # ax1.plot(lmbda[hopf_list], val_norm[hopf_list], "o", color="red")
    # ax1.set_xlim(0.0, 4.0)
    # ax1.set_ylim(0.0, 14.0)
    # fig.savefig("test_bratu.png", type="PNG", bbox_inches="tight", transparent=False)


if __name__ == "__main__":
    test_bratu()
