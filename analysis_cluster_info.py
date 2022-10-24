import numpy as np
import os
import matplotlib.pyplot as plt

from sificc_lib import Preprocessing
from sificc_lib.root_files import root_files
from uproot_methods.classes.TVector3 import TVector3

########################################################################################################################

dir_main = os.getcwd()
# update plot font size
plt.rcParams.update({'font.size': 12})

# number of statistics used for analysis
n = 100000

########################################################################################################################

preprocessing_0mm = Preprocessing(dir_main + root_files.optimized_0mm_local)
preprocessing_5mm = Preprocessing(dir_main + root_files.optimized_5mm_local)
print("loaded file: ", root_files.optimized_0mm_local)
print("loaded file: ", root_files.optimized_5mm_local)

# predefine lists
n_clusters_total = [[], []]
n_clusters_scatterer = [[], []]
n_clusters_absorber = [[], []]

# iterate through 0mm dataset
for event in preprocessing_0mm.iterate_events(n=n):
    # skip all none distributed events
    if not event.is_distributed:
        continue

    # read number of clusters per event and per module
    n_clusters_total[0].append(len(event.RecoClusterPosition))
    idx_scatterer, idx_absorber = event.sort_clusters_by_module()
    n_clusters_scatterer[0].append(len(idx_scatterer))
    n_clusters_absorber[0].append(len(idx_absorber))


bins_cluster = np.arange(-0.5, 10.5, 1.0)
hist_count, _ = np.histogram(n_clusters_total[0], bins=bins_cluster)

# plot distribution number of cluster per event
plt.figure()
plt.bar(bins_cluster[:-1], hist_count, align="edge", color="blue")
plt.show()