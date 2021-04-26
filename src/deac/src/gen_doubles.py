#!/bin/env python3
import numpy as np
data = np.arange(12,dtype=np.double)
data = np.reshape(data,(3,4))
fn = "double_data.dat"
with open(fn, mode='wb') as f:
    data.tofile(f)
