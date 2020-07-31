import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def parseCsvFile(filename, column_keys=None):
    column_index = {}  # mapping, key=column_key, value=corresponding column index
    data = {}  # dict of data, key=column_key, value=data list (floats)
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        line_i = 0  # line index
        for row in csvreader:
            if line_i == 0:
                # Headers. Find interesting columns
                headers = row
                # prepare structure for all columns we want
                if column_keys is None:
                    # grab all data in file
                    column_keys_we_want = [elt.lower() for elt in headers]
                else:
                    # grab only requested data from file
                    assert type(column_keys) == type([])
                    column_keys_we_want = column_keys
                for column_key in column_keys_we_want:
                    data[column_key] = []
                for column_i, elt in enumerate(headers):
                    elt_lower = elt.lower()
                    if elt_lower in column_keys_we_want:
                        column_index[elt_lower] = column_i
                line_i += 1
                continue
            # Data line
            if len(row) < len(headers):
                break  # finished reading all data
            for column_key in column_keys_we_want:
                data[column_key].append(float(row[column_index[column_key]]))
            line_i += 1
            continue  # go to next data line
    return data


def plot_continuation_results(filename):
    """

"""
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
    data = parseCsvFile(filename)
    # Unpack data
    lmbda = np.asarray(data["lambda"])
    u_norm = np.asarray(data["u_norm"])
    stability = np.asarray(data["stability"], dtype=bool)
    oscillation = np.asarray(data["oscillation"], dtype=bool)
    saddle = np.asarray(data["saddle"], dtype=bool)
    hopf = np.asarray(data["hopf"], dtype=bool)

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
