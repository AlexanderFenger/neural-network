import numpy as np
import os
import matplotlib.pyplot as plt

from sificc_lib import Preprocessing
from sificc_lib.root_files import root_files
from sificc_lib import utilities
from uproot_methods.classes.TVector3 import TVector3

########################################################################################################################

dir_main = os.getcwd()
# update plot font size
plt.rcParams.update({'font.size': 12})

########################################################################################################################

preprocessing_0mm = Preprocessing(dir_main + root_files.optimized_0mm_local)
preprocessing_5mm = Preprocessing(dir_main + root_files.optimized_5mm_local)
print("loaded file: ", root_files.optimized_0mm_local)
print("loaded file: ", root_files.optimized_5mm_local)


# utilities.print_event_summary(preprocessing_0mm, 30)


def process_data(preprocessing: Preprocessing):
    data = []

    # number of statistics used for analysis
    n = 10000

    # iterate through 0mm dataset
    for event in preprocessing.iterate_events(n=n):
        # skip all none distributed events
        if not event.is_distributed:
            continue

        # create empty list to collect event data
        """
        Fromat:
        0: EventID, 
        1: n-cluster, 
        2: n-cluster-scatterer, 
        3: n-cluster-absorber,
        4: tag ideal compton
        5: tag CB
        6: cluster-id scatterer highest energy
        7: cluster-id absorber highest energy
        8: cluster-id matching e-position
        9: cluster-id matching p-position
        """
        event_data = []

        # add event id
        event_data.append(event.EventNumber)

        # sort cluster by module (sorting based on energy, return based on array idx)
        idx_scatterer, idx_absorber = event.sort_clusters_by_module()

        # read number of clusters per event and per module
        event_data.append(len(event.RecoClusterPosition))
        event_data.append(len(idx_scatterer))
        event_data.append(len(idx_absorber))

        # ideal compton tag
        event_data.append(event.is_compton_ideal * 1)

        # Cut-Based selection tag
        event_data.append(event.Identified * 1)

        # compare the highest energy cluster in scatterer and absorber and only
        # count the higher one
        event_data.append((idx_scatterer[0] - len(idx_scatterer) + 1) * (-1))
        event_data.append((idx_absorber[0] - len(idx_absorber) + 1) * (-1))

        # argmatch clustering for electron and photon
        if event.is_compton_ideal:
            event_data.append(event.argmatch_cluster(event.MCPosition_e_first))
            event_data.append(event.argmatch_cluster(event.MCPosition_p_first))
            # correct scatterer indexing
            if event_data[8] != -1:
                event_data[8] = (event_data[8] - (len(idx_scatterer) + len(idx_absorber)) + 1) * (-1)

            # correct absorber indexing
            if event_data[9] != -1:
                event_data[9] = (event_data[9] - len(idx_absorber) + 1) * (-1)
        else:
            event_data.append(-2)
            event_data.append(-2)

        # end
        data.append(event_data)

    return np.array(data)


# grab data
data_0mm = process_data(preprocessing_0mm)
data_5mm = process_data(preprocessing_5mm)

########################################################################################################################

# define bin edges for cluster counting
bins_cluster = np.arange(-0.5, 14.5, 1.0)
# get total number of entries for normalization
n_0mm = np.sum(data_0mm[:, 1])
n_5mm = np.sum(data_5mm[:, 1])

###########################################################################
# plot distribution number of cluster total
hist_0mm_total, _ = np.histogram(data_0mm[:, 1], bins=bins_cluster)
hist_5mm_total, _ = np.histogram(data_5mm[:, 1], bins=bins_cluster)

plt.figure()
plt.title("Distribution cluster counts total")
plt.xlabel("# of clusters")
plt.ylabel("counts (normalized)")
plt.xlim(0, 15)
plt.xticks(bins_cluster + 0.5)
plt.bar(bins_cluster[:-1] + 0.30, hist_0mm_total / np.sum(n_0mm), fill=True, width=0.4, align="center",
        color="black", label="0mm", alpha=1.0)
plt.bar(bins_cluster[:-1] + 0.70, hist_5mm_total / np.sum(n_5mm), fill=True, width=0.4, align="center",
        color="red", label="5mm", alpha=1.0)
plt.legend()
plt.show()

##########################################################################
# plot distribution number of cluster per event for scatterer and absorber
bins_cluster = np.arange(-0.5, 10.5, 1.0)
hist_0mm_scatterer, _ = np.histogram(data_0mm[:, 2], bins=bins_cluster)
hist_0mm_absorber, _ = np.histogram(data_0mm[:, 3], bins=bins_cluster)

hist_0mm_scatterer_ic, _ = np.histogram([data_0mm[i, 2] for i in range(len(data_0mm[:, 2])) if data_0mm[i, 4] == 1],
                                        bins=bins_cluster)
hist_0mm_absorber_ic, _ = np.histogram([data_0mm[i, 3] for i in range(len(data_0mm[:, 3])) if data_0mm[i, 4] == 1],
                                       bins=bins_cluster)

plt.figure()
plt.title("Distribution cluster per module (0mm)")
plt.xlabel("# of clusters")
plt.ylabel("counts (normalized per module)")
plt.xlim(-0.5, 12)
plt.xticks(bins_cluster + 0.5)
plt.bar(bins_cluster[:-1] + 0.30, hist_0mm_scatterer / np.sum(hist_0mm_scatterer), width=0.4, align="center",
        color="blue", alpha=0.8, label="scatterer")
plt.bar(bins_cluster[:-1] + 0.70, hist_0mm_absorber / np.sum(hist_0mm_absorber), width=0.4, align="center",
        color="orange", alpha=0.8, label="absorber")
plt.plot(bins_cluster[1:-1] + 0.30, hist_0mm_scatterer_ic[1:] / np.sum(hist_0mm_scatterer_ic), color="darkblue",
         linestyle="-.", label="scatterer ideal compton", marker="x")
plt.plot(bins_cluster[1:-1] + 0.70, hist_0mm_absorber_ic[1:] / np.sum(hist_0mm_absorber_ic), color="darkorange",
         linestyle="--", label="absorber ideal compton", marker="o")
plt.legend()
plt.show()

###########################################################################
# plot distribution number of cluster per CB event for scatterer
hist_0mm_scatterer, _ = np.histogram([data_0mm[i, 2] for i in range(len(data_0mm[:, 2])) if data_0mm[i, 5] != 0],
                                     bins=bins_cluster)
hist_0mm_absorber, _ = np.histogram([data_0mm[i, 3] for i in range(len(data_0mm[:, 3])) if data_0mm[i, 5] != 0],
                                    bins=bins_cluster)

plt.figure()
plt.title("Distribution cluster per module (0mm, CB-selected)")
plt.xlabel("# of clusters")
plt.ylabel("counts (normalized per module)")
plt.xlim(-0.5, 11)
plt.xticks(bins_cluster + 0.5)
plt.bar(bins_cluster[:-1] + 0.30, hist_0mm_scatterer / np.sum(hist_0mm_scatterer), width=0.4, align="center",
        color="blue", alpha=1.0, label="scatterer")
plt.bar(bins_cluster[:-1] + 0.70, hist_0mm_absorber / np.sum(hist_0mm_absorber), width=0.4, align="center",
        color="orange", alpha=1.0, label="absorber")
plt.legend()
plt.show()

###########################################################################
# plot distribution cluster idx with the highest energy
# competitive for both modules
hist_0mm_scatterer, _ = np.histogram(data_0mm[:, 6], bins=bins_cluster)
hist_0mm_absorber, _ = np.histogram(data_0mm[:, 7], bins=bins_cluster)

hist_0mm_scatterer_ic, _ = np.histogram([data_0mm[i, 6] for i in range(len(data_0mm[:, 6])) if data_0mm[i, 4] != 0],
                                        bins=bins_cluster)
hist_0mm_absorber_ic, _ = np.histogram([data_0mm[i, 7] for i in range(len(data_0mm[:, 7])) if data_0mm[i, 4] != 0],
                                       bins=bins_cluster)

plt.figure()
plt.title("Distribution highest energy cluster (0mm)")
plt.xlabel("cluster idx")
plt.ylabel("counts (normalized per module)")
plt.xlim(-0.5, 10.5)
plt.xticks(bins_cluster + 0.5)
plt.bar(bins_cluster[:-1] + 0.30, hist_0mm_scatterer / (np.sum(hist_0mm_scatterer)),
        width=0.4, align="center", color="blue", alpha=0.8, label="scatterer")
plt.bar(bins_cluster[:-1] + 0.70, hist_0mm_absorber / (np.sum(hist_0mm_absorber)),
        width=0.4, align="center", color="orange", alpha=0.8, label="absorber")
plt.plot(bins_cluster[:-1] + 0.30,
         hist_0mm_scatterer_ic / (np.sum(hist_0mm_scatterer_ic))
         , color="darkblue", linestyle="-.", label="scatterer ideal compton", marker="x")
plt.plot(bins_cluster[:-1] + 0.70,
         hist_0mm_absorber_ic / (np.sum(hist_0mm_absorber_ic))
         , color="darkorange", linestyle="--", label="absorber ideal compton", marker="o")
plt.legend()
plt.show()

###########################################################################
# plot distribution cluster idx matching e and p position
bins_argmatch = np.arange(-1.5, 10.5, 1.0)
hist_0mm_e_argmatch, _ = np.histogram(data_0mm[:, 8], bins=bins_argmatch)
hist_0mm_p_argmatch, _ = np.histogram(data_0mm[:, 9], bins=bins_argmatch)

plt.figure()
plt.title("Distribution argmatch (0mm, ideal compton)")
plt.xlabel("cluster idx")
plt.ylabel("counts (normalized)")
plt.xlim(-1.5, 10)
plt.xticks(bins_argmatch + 0.5)
plt.bar(bins_argmatch[:-1] + 0.30, hist_0mm_e_argmatch / np.sum(hist_0mm_e_argmatch), width=0.4, align="center",
        color="blue", alpha=1.0, label="e pos")
plt.bar(bins_argmatch[:-1] + 0.70, hist_0mm_p_argmatch / np.sum(hist_0mm_p_argmatch), width=0.4, align="center",
        color="orange", alpha=1.0, label="p pos")
plt.legend()
plt.show()
