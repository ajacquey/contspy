import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def split_stability_results(u_norm, stability, oscillation):
    """

"""
    u_norm_stable_reg = np.ones_like(u_norm) * np.NaN
    u_norm_unstable_reg = np.ones_like(u_norm) * np.NaN
    u_norm_stable_osc = np.ones_like(u_norm) * np.NaN
    u_norm_unstable_osc = np.ones_like(u_norm) * np.NaN

    u_norm_stable_reg[(stability) & ~oscillation] = u_norm[(stability) & ~oscillation]
    u_norm_unstable_reg[~stability & ~oscillation] = u_norm[~stability & ~oscillation]
    u_norm_stable_osc[stability & oscillation] = u_norm[stability & oscillation]
    u_norm_unstable_osc[~stability & oscillation] = u_norm[~stability & oscillation]

    return (
        u_norm_stable_reg,
        u_norm_stable_osc,
        u_norm_unstable_reg,
        u_norm_unstable_osc,
    )


def get_bifurcation_points(lmbda, u_norm, saddle, hopf):
    """

"""
    lmbda_saddle = lmbda[saddle]
    u_norm_saddle = u_norm[saddle]
    lmbda_hopf = lmbda[hopf]
    u_norm_hopf = u_norm[hopf]

    return lmbda_saddle, lmbda_hopf, u_norm_saddle, u_norm_hopf


def plot_continuation_results(filename, nvar=1):
    """

"""
    # Path to script
    script_file = os.path.abspath(sys.argv[0]).split("/")[-1]
    filename = os.path.abspath(sys.argv[0]).replace(script_file, filename)
    # Check if file exists
    if not os.path.exists(filename):
        raise Exception(
            "File",
            filename,
            "does not exits, please check the name and path of the given file!",
        )

    print()
    print("Reading data stored in file", filename, "...")

    # Load data
    cols = [k + 1 for k in range(nvar)]
    lmbda = np.loadtxt(
        filename, dtype=float, delimiter=",", skiprows=1, unpack=True, usecols=[0]
    )
    u_norm = np.loadtxt(
        filename, dtype=float, delimiter=",", skiprows=1, unpack=False, usecols=cols
    )
    cols = [k + nvar + 1 for k in range(4)]
    stability, oscillation, saddle, hopf = np.loadtxt(
        filename, dtype=bool, delimiter=",", skiprows=1, unpack=True, usecols=cols,
    )

    if nvar > 1:
        u_norm = list(u_norm.T)

        u_norm_stable_reg = [None] * nvar
        u_norm_stable_osc = [None] * nvar
        u_norm_unstable_reg = [None] * nvar
        u_norm_unstable_osc = [None] * nvar
        u_norm_saddle = [None] * nvar
        u_norm_hopf = [None] * nvar

        for k in range(nvar):
            (
                u_norm_stable_reg[k],
                u_norm_stable_osc[k],
                u_norm_unstable_reg[k],
                u_norm_unstable_osc[k],
            ) = split_stability_results(u_norm[k], stability, oscillation)
            (
                lmbda_saddle,
                lmbda_hopf,
                u_norm_saddle[k],
                u_norm_hopf[k],
            ) = get_bifurcation_points(lmbda, u_norm[k], saddle, hopf)
    else:
        (
            u_norm_stable_reg,
            u_norm_stable_osc,
            u_norm_unstable_reg,
            u_norm_unstable_osc,
        ) = split_stability_results(u_norm, stability, oscillation)
        lmbda_saddle, lmbda_hopf, u_norm_saddle, u_norm_hopf = get_bifurcation_points(
            lmbda, u_norm, saddle, hopf
        )

    # Figure name
    fig_filename = os.path.splitext(os.path.abspath(filename))[0] + ".png"

    # Plot
    plt.rc("text", usetex=True)
    fig, axes = plt.subplots(1, nvar)
    if nvar > 1:
        for k in range(nvar):
            initialize_plot(axes[k], r"$||u_{" + str(k) + r"}||_{\infty}$")
            plot_continuation_lines(
                axes[k],
                lmbda,
                u_norm_stable_reg[k],
                u_norm_stable_osc[k],
                u_norm_unstable_reg[k],
                u_norm_unstable_osc[k],
            )
            plot_continuation_points(
                axes[k], lmbda_saddle, u_norm_saddle[k], lmbda_hopf, u_norm_hopf[k]
            )
    else:
        initialize_plot(axes, r"$||u||_{\infty}$")
        plot_continuation_lines(
            axes,
            lmbda,
            u_norm_stable_reg,
            u_norm_stable_osc,
            u_norm_unstable_reg,
            u_norm_unstable_osc,
        )
        plot_continuation_points(
            axes, lmbda_saddle, u_norm_saddle, lmbda_hopf, u_norm_hopf
        )

    # ax.set_xlim(0.0, 1.2)
    # ax.set_ylim(0.0, 15.0)
    fig.set_size_inches(8 * nvar, 6)
    fig.savefig(fig_filename, type="PNG", bbox_inches="tight", transparent=False)
    print("Plot of continuation results in file", fig_filename, "completed!")

    return None


def initialize_plot(ax, label):
    """

"""
    ax.set_xlabel(r"$\lambda$", fontsize=18)
    ax.set_ylabel(label, fontsize=18)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid()
    return None


def plot_continuation_lines(
    ax, lmbda, u_stable_reg, u_stable_osc, u_unstable_reg, u_unstable_osc
):
    """
    Plot the stable/unstable and regular/oscillatory solutions for the contiuation
    INPUT:
    ax: matplotlib axes
    lmbda: array containing the bifurcation parameter values
    u_stable_reg: infinite norm of the variable for stable and regular solutions
    u_stable_osc: infinite norm of the variable for stable and oscillatory solutions
    u_unstable_reg: infinite norm of the variable for unstable and regular solutions
    u_unstable_osc: infinite norm of the variable for unstable and oscillatory solutions
"""
    ax.plot(lmbda, u_stable_reg, "-", color="blue")
    ax.plot(lmbda, u_unstable_reg, "--", color="blue")
    ax.plot(lmbda, u_stable_osc, "-", color="red")
    ax.plot(lmbda, u_unstable_osc, "--", color="red")

    return None


def plot_continuation_points(ax, lmbda_saddle, u_saddle, lmbda_hopf, u_hopf):
    """
    Plot the saddle and Hopf bifurcation points from the continuation results
    INPUT
    ax: matplotlib axes
    lmbda_saddle: array of the bifurcation parameter values corresponding to saddle points
    u_saddle: array of the infinite norms of the variable corresponding to saddle points
    lmbda_hopf: array of the bifurcation parameter values corresponding to Hopf points
    u_hopf: array of the infinite norms of the variable corresponding to Hopf points
"""
    ax.plot(lmbda_saddle, u_saddle, "o", color="blue")
    ax.plot(lmbda_hopf, u_hopf, "o", color="red")

    return None
