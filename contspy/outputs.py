import csv
import os
import sys

import numpy as np


def initialize_output(filename, headers, output_steps):
    """
    Initialize the output of continuation in a CSV file
"""
    output_fname = get_output_filename(filename)
    with open(output_fname, "w", newline="") as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(headers)

    if output_steps:
        # Here we need to create the folder 'steps' if it does not exist
        csv_filename = output_fname.split("/")[-1]
        path = output_fname.replace(csv_filename, "steps/")
        isdir = os.path.isdir(path)

        if not isdir:
            try:
                os.mkdir(path)
            except OSError:
                print("Creation of the directory %s failed" % path)

        output_steps_fname = path + csv_filename.split(".")[0] + "_"
    else:
        output_steps_fname = ""

    return output_fname, output_steps_fname


def write_output(
    k,
    output_fname,
    output_steps_fname,
    x,
    u,
    lmbda,
    stability,
    oscillation,
    saddle,
    hopf,
):
    """
    Output continuation step and spectral step in a CSV file
"""
    # Continuation
    results = [
        lmbda,
        np.linalg.norm(u, np.inf),
        int(stability),
        int(oscillation),
        int(saddle),
        int(hopf),
    ]
    with open(output_fname, "a+", newline="") as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(results)

    # Spectral
    if bool(output_steps_fname):  # string not empty
        filename = output_steps_fname + str(k) + ".csv"
        u = np.concatenate([[0.0], u, [0.0]])
        data = np.column_stack((np.flip(x), u))
        np.savetxt(
            filename, data, delimiter=",", fmt="%1.4e", header="x,u", comments=""
        )
    return None


def get_output_filename(filename):
    """
  Get the file name for the output CSV file
"""
    if filename is None:
        script_fname = os.path.splitext(os.path.abspath(sys.argv[0]))[0]
        output_fname = script_fname + "_out.csv"
    else:
        path_to_script = os.path.abspath(os.path.dirname(sys.argv[0]))
        output_fname = path_to_script + "/" + filename + "_out.csv"

    return output_fname
