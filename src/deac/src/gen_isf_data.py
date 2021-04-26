#!/bin/env python3
import numpy as np
data = np.arange(12,dtype=np.double)
data_isf = np.zeros((12,3),dtype=np.double)
data_isf[:,0] = data
data_isf[:,1] = data + 10
data_isf[:,2] = data + 100
fn = "isf_data.dat"
with open(fn, mode='wb') as f:
    data_isf.tofile(f)
