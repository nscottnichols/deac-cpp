import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = "/path/to/deac_results_dir"
figure_fn = "dsf_average.pdf"

#get deac data files
dsf_files = {}
frequency_files = {}
for f in os.listdir(data_dir):
    key = f.split(".")[0].split("_")[-1]
    value = os.path.join(data_dir,f)
    if "dsf" in f:
        dsf_files[key] = value
    if "frequency" in f:
        frequency_files[key] = value
keys = np.sort(np.array(list(dsf_files.keys())))

#get average of dynamic structure factor
frequency_avg = np.fromfile(frequency_files[keys[0]], dtype=np.double)
dsf_avg = np.fromfile(dsf_files[keys[0]], dtype=np.double)
for key in keys[1:]:
    dsf_fn = dsf_files[key]
    dsf = np.fromfile(dsf_fn, dtype=np.double)
    dsf_avg += dsf
    del dsf
dsf_avg /= keys.shape[0]

#save plot
fig, ax = plt.subplots(figsize=(8,4.5), dpi=120, constrained_layout=True)
ax.plot(frequency_avg, dsf_avg)
ax.set_xlabel("Frequency")
ax.set_ylabel("Dynamic Structure Factor")
fig.savefig(figure_fn)
