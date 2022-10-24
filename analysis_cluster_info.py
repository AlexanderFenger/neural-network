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
n = 10000

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

# iterate through 5mm dataset
for event in preprocessing_5mm.iterate_events(n=n):
    # skip all none distributed events
    if not event.is_distributed:
        continue

    # read number of clusters per event and per module
    n_clusters_total[1].append(len(event.RecoClusterPosition))
    idx_scatterer, idx_absorber = event.sort_clusters_by_module()
    n_clusters_scatterer[1].append(len(idx_scatterer))
    n_clusters_absorber[1].append(len(idx_absorber))

########################################################################################################################

bins_cluster = np.arange(-0.5, 10.5, 1.0)

# plot distribution number of cluster per event
hist_0mm, _ = np.histogram(n_clusters_total[0], bins=bins_cluster)
hist_5mm, _ = np.histogram(n_clusters_total[1], bins=bins_cluster)

plt.figure()
plt.title("Distribution cluster counts (total)")
plt.xlabel("# of clusters")
plt.ylabel("counts")
plt.xlim(0, 11)
plt.xticks(bins_cluster + 0.5)
plt.bar(bins_cluster[:-1] + 0.30, hist_0mm / np.sum(hist_0mm), width=0.4, align="center", color="black", alpha=1.0,
        label="0mm")
plt.bar(bins_cluster[:-1] + 0.70, hist_5mm / np.sum(hist_5mm), width=0.4, align="center", color="red", alpha=1.0,
        label="5mm")
plt.legend()
plt.show()

# plot distribution number of cluster per event for scatterer
hist_0mm, _ = np.histogram(n_clusters_scatterer[0], bins=bins_cluster)
hist_5mm, _ = np.histogram(n_clusters_scatterer[1], bins=bins_cluster)

plt.figure()
plt.title("Distribution cluster counts (scatterer)")
plt.xlabel("# of clusters")
plt.ylabel("counts")
plt.xlim(-0.5, 11)
plt.xticks(bins_cluster + 0.5)
plt.bar(bins_cluster[:-1] + 0.30, hist_0mm / np.sum(hist_0mm), width=0.4, align="center", color="black", alpha=1.0,
        label="0mm")
plt.bar(bins_cluster[:-1] + 0.70, hist_5mm / np.sum(hist_5mm), width=0.4, align="center", color="red", alpha=1.0,
        label="5mm")
plt.legend()
plt.show()

# plot distribution number of cluster per event for absorber
hist_0mm, _ = np.histogram(n_clusters_absorber[0], bins=bins_cluster)
hist_5mm, _ = np.histogram(n_clusters_absorber[1], bins=bins_cluster)

plt.figure()
plt.title("Distribution cluster counts (absorber)")
plt.xlabel("# of clusters")
plt.ylabel("counts")
plt.xlim(-0.5, 11)
plt.xticks(bins_cluster + 0.5)
plt.bar(bins_cluster[:-1] + 0.30, hist_0mm / np.sum(hist_0mm), width=0.4, align="center", color="black", alpha=1.0,
        label="0mm")
plt.bar(bins_cluster[:-1] + 0.70, hist_5mm / np.sum(hist_5mm), width=0.4, align="center", color="red", alpha=1.0,
        label="5mm")
plt.legend()
plt.show()
