#!/bin/env python3
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Convert isf NPZ file to binary file.')
parser.add_argument('isf_file_in', type=str,
                    help='npz isf file in location')
args = parser.parse_args()
isf_fn = args.isf_file_in
isf_npz_data = np.load(isf_fn)
print("imaginary_time: ")
print(isf_npz_data["tau"].astype(np.double))
print("isf: ")
print(isf_npz_data["isf"].astype(np.double))
print("isf_error: ")
print(isf_npz_data["error"].astype(np.double))
