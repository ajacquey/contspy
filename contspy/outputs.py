import os
import sys

import numpy as np


def initialize_output(filename, headers, output_steps):
    """
    Initialize the output of continuation in a CSV file
"""
    output_fname = get_output_filename(filename)
    np.savetxt(
        output_fname, [np.asarray(headers)], delimiter=",", fmt="%s", comments=""
    )

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

        output_steps_fname = path + ".".join(csv_filename.split(".")[:-1]) + "_"
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
    nvar,
    stability,
    oscillation,
    saddle,
    hopf,
):
    """
    Output continuation step and spectral step in a CSV file
"""
    # Continuation
    if nvar > 1:
        uvars = np.split(u, nvar)
        results_u = [np.linalg.norm(uvar, np.inf) for uvar in uvars]
        results = [
            lmbda,
            int(stability),
            int(oscillation),
            int(saddle),
            int(hopf),
        ]
        results[1:1] = results_u
    else:
        results = [
            lmbda,
            np.linalg.norm(u, np.inf),
            int(stability),
            int(oscillation),
            int(saddle),
            int(hopf),
        ]

    fmt = ["%1.4e", "%1i", "%1i", "%1i", "%1i"]
    fmt_var = ["%1.4e"] * nvar
    fmt[1:1] = fmt_var
    with open(output_fname, "a+", newline="") as write_obj:
        np.savetxt(
            write_obj, [results], fmt=fmt, comments="", delimiter=",",
        )

    # Spectral
    if bool(output_steps_fname):  # string not empty
        filename = output_steps_fname + str(k) + ".csv"
        if nvar > 1:
            uvars = np.split(u, nvar)
            uvars = [np.concatenate([[0.0], uvar, [0.0]]) for uvar in uvars]
            uvars = np.array(uvars)
            header = ["x"]
            header_var = ["u" + str(int(k)) for k in range(nvar)]
            header[1:1] = header_var
            header = ",".join(header)
            data = np.column_stack((np.flip(x), uvars.transpose()))
        else:
            u = np.concatenate([[0.0], u, [0.0]])
            header = "x,u"
            data = np.column_stack((np.flip(x), u))
        np.savetxt(
            filename, data, delimiter=",", fmt="%1.4e", header=header, comments=""
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
