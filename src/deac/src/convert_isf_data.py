#!/bin/env python3
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Convert isf NPZ file to binary file.')
parser.add_argument('isf_file_in', type=str,
                    help='npz isf file in location')
parser.add_argument('isf_file_out', type=str,
                    help='binary isf file out location')
args = parser.parse_args()
isf_fn = args.isf_file_in
isf_npz_data = np.load(isf_fn)
data_isf = np.zeros((3,isf_npz_data["tau"].shape[0]),dtype=np.double)
data_isf[0,:] = isf_npz_data["tau"].astype(np.double)
data_isf[1,:] = isf_npz_data["isf"].astype(np.double)
data_isf[2,:] = isf_npz_data["error"].astype(np.double)
fn = args.isf_file_out
with open(fn, mode='wb') as f:
    data_isf.tofile(f)
