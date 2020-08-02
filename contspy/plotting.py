import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def plot_continuation_results(filename):
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
    lmbda, u_norm = np.loadtxt(
        filename, dtype=float, delimiter=",", skiprows=1, unpack=True, usecols=[0, 1]
    )
    stability, oscillation, saddle, hopf = np.loadtxt(
        filename,
        dtype=bool,
        delimiter=",",
        skiprows=1,
        unpack=True,
        usecols=[2, 3, 4, 5],
    )

    # Handle data to prepare arrays to plot
    u_norm_stable_reg = np.ones_like(u_norm) * np.NaN
    u_norm_instable_reg = np.ones_like(u_norm) * np.NaN
    u_norm_stable_osc = np.ones_like(u_norm) * np.NaN
    u_norm_instable_osc = np.ones_like(u_norm) * np.NaN

    u_norm_stable_reg[(stability) & ~oscillation] = u_norm[(stability) & ~oscillation]
    u_norm_instable_reg[~stability & ~oscillation] = u_norm[~stability & ~oscillation]
    u_norm_stable_osc[stability & oscillation] = u_norm[stability & oscillation]
    u_norm_instable_osc[~stability & oscillation] = u_norm[~stability & oscillation]

    # Handle dara to prepare saddle and hopf points to plot
    lmbda_saddle = lmbda[saddle]
    u_norm_saddle = u_norm[saddle]
    lmbda_hopf = lmbda[hopf]
    u_norm_hopf = u_norm[hopf]

    # Figure name
    fig_filename = os.path.splitext(os.path.abspath(filename))[0] + ".png"

    # Plot
    plt.rc("text", usetex=True)
    fig, ax = plt.subplots(1)
    ax.set_xlabel(r"$\lambda$", fontsize=18)
    ax.set_ylabel(r"$||u||_{\infty}$", fontsize=18)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid()

    ax.plot(lmbda, u_norm_stable_reg, "-", color="blue")
    ax.plot(lmbda, u_norm_instable_reg, "--", color="blue")
    ax.plot(lmbda, u_norm_stable_osc, "-", color="red")
    ax.plot(lmbda, u_norm_instable_osc, "--", color="red")
    ax.plot(lmbda_saddle, u_norm_saddle, "o", color="blue")
    ax.plot(lmbda_hopf, u_norm_hopf, "o", color="red")
    # ax.set_xlim(0.0, 1.2)
    # ax.set_ylim(0.0, 15.0)
    fig.set_size_inches(8, 6)
    fig.savefig(fig_filename, type="PNG", bbox_inches="tight", transparent=False)
    print("Plot of continuation results in file", fig_filename, "completed!")
