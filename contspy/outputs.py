import csv
import os
import sys

import numpy as np


def initialize_output(filename, headers):
    """
    Initialize the output of continuation in a CSV file
"""
    output_fname = get_output_filename(filename)
    with open(output_fname, "w", newline="") as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(headers)
    return output_fname


def write_output(output_fname, u, lmbda, stability, oscillation, saddle, hopf):
    """
    Output continuation step in a CSV file
"""
    results = [lmbda, np.linalg.norm(u, np.inf), stability, oscillation, saddle, hopf]
    with open(output_fname, "a+", newline="") as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(results)


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
