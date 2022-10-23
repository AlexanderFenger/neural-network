import numpy as np
import os
import matplotlib.pyplot as plt
# test for extracting event information from root files
from sificc_lib_awal import Simulation
from sificc_lib_awal import utils
from sificc_lib.root_files import root_files
from uproot_methods.classes.TVector3 import TVector3

########################################################################################################################

dir_main = os.getcwd()
# update plot font size
plt.rcParams.update({'font.size': 12})


########################################################################################################################


def readout_awal_target():
    data_0mm = np.load(dir_main + "/data/" + "optimized_0mm_training" + ".npz")
    data_5mm = np.load(dir_main + "/data/" + "optimized_5mm_training" + ".npz")

    targets_0mm = data_0mm["targets"]
    targets_5mm = data_5mm["targets"]

    # event ratio
    print("Ideal Compton event ratio:")
    print("0mm: {:.3f}% ({} / {})".format(np.sum(targets_0mm[:, 0]) / targets_0mm.shape[0], np.sum(targets_0mm[:, 0]),
                                          targets_0mm.shape[0]))
    print("5mm: {:.3f}% ({} / {})".format(np.sum(targets_5mm[:, 0]) / targets_5mm.shape[0], np.sum(targets_5mm[:, 0]),
                                          targets_5mm.shape[0]))

    # plot electron position z distribution
    e_posz_0mm = [targets_0mm[i, 5] for i in range(targets_0mm.shape[0]) if targets_0mm[i, 0] != 0]
    e_posz_5mm = [targets_5mm[i, 5] for i in range(targets_5mm.shape[0]) if targets_5mm[i, 0] != 0]
    bins = np.arange(-98.8 / 2 - 1.3 / 2, 98.8 / 2 + 1.3 / 2, 1.3)
    hist_counts_0mm, hist_bins = np.histogram(e_posz_0mm, bins=bins)
    hist_counts_5mm, hist_bins = np.histogram(e_posz_5mm, bins=bins)

    plt.figure()
    plt.title("Target electron position (Ideal compton events)")
    plt.xlabel("Position [mm]")
    plt.ylabel("counts")
    plt.hist(e_posz_0mm, bins=bins, histtype=u"step", color="black", label="0mm", density=True)
    plt.hist(e_posz_5mm, bins=bins, histtype=u"step", color="red", label="5mm", density=True)
    plt.errorbar(hist_bins[1:] - 1.3 / 2, hist_counts_0mm / (np.sum(hist_counts_0mm) * 1.3),
                 np.sqrt(hist_counts_0mm) / (np.sum(hist_counts_0mm) * 1.3), color="black", fmt=".")
    plt.errorbar(hist_bins[1:] - 1.3 / 2, hist_counts_5mm / (np.sum(hist_counts_5mm) * 1.3),
                 np.sqrt(hist_counts_5mm) / (np.sum(hist_counts_5mm) * 1.3), color="red", fmt=".")
    plt.vlines(x=np.mean(e_posz_0mm), ymin=0, ymax=0.015, linestyles="--", color="black",
               label=r"$\mu$ = {:.1f}".format(np.mean(e_posz_0mm)))
    plt.vlines(x=np.mean(e_posz_5mm), ymin=0, ymax=0.015, linestyles="--", color="red",
               label=r"$\mu$ = {:.1f}".format(np.mean(e_posz_5mm)))
    plt.legend()
    plt.grid()
    plt.show()

    # plot photon position z distribution
    p_posz_0mm = [targets_0mm[i, 8] for i in range(targets_0mm.shape[0]) if targets_0mm[i, 0] != 0]
    p_posz_5mm = [targets_5mm[i, 8] for i in range(targets_5mm.shape[0]) if targets_5mm[i, 0] != 0]
    bins = np.arange(-98.8 / 2 - 1.3 / 2, 98.8 / 2 + 1.3 / 2, 1.3)
    hist_counts_0mm, hist_bins = np.histogram(p_posz_0mm, bins=bins)
    hist_counts_5mm, hist_bins = np.histogram(p_posz_5mm, bins=bins)

    plt.figure()
    plt.title("Target photon position (Ideal compton events)")
    plt.xlabel("Position [mm]")
    plt.ylabel("counts")
    plt.hist(p_posz_0mm, bins=bins, histtype=u"step", color="black", label="0mm", density=True)
    plt.hist(p_posz_5mm, bins=bins, histtype=u"step", color="red", label="5mm", density=True)
    plt.errorbar(hist_bins[1:] - 1.3 / 2, hist_counts_0mm / (np.sum(hist_counts_0mm) * 1.3),
                 np.sqrt(hist_counts_0mm) / (np.sum(hist_counts_0mm) * 1.3), color="black", fmt=".")
    plt.errorbar(hist_bins[1:] - 1.3 / 2, hist_counts_5mm / (np.sum(hist_counts_5mm) * 1.3),
                 np.sqrt(hist_counts_5mm) / (np.sum(hist_counts_5mm) * 1.3), color="red", fmt=".")
    plt.vlines(x=np.mean(p_posz_0mm), ymin=0, ymax=0.015, linestyles="--", color="black",
               label=r"$\mu$ = {:.1f}".format(np.mean(p_posz_0mm)))
    plt.vlines(x=np.mean(p_posz_5mm), ymin=0, ymax=0.015, linestyles="--", color="red",
               label=r"$\mu$ = {:.1f}".format(np.mean(p_posz_5mm)))
    plt.legend()
    plt.grid()
    plt.show()

    # plot angular distribution of the scatter direction
    theta_0mm = []
    theta_5mm = []

    for i in range(targets_0mm.shape[0]):
        if targets_0mm[i, 0] != 0:
            vec = TVector3(targets_0mm[i, 6] - targets_0mm[i, 3],
                           targets_0mm[i, 7] - targets_0mm[i, 4],
                           targets_0mm[i, 8] - targets_0mm[i, 5])
            theta_0mm.append(vec.phi)

    for i in range(targets_5mm.shape[0]):
        if targets_5mm[i, 0] != 0:
            vec = TVector3(targets_5mm[i, 6] - targets_5mm[i, 3],
                           targets_5mm[i, 7] - targets_5mm[i, 4],
                           targets_5mm[i, 8] - targets_5mm[i, 5])
            theta_5mm.append(vec.phi)

    # plot photon position z distribution
    bins = np.arange(-np.pi, np.pi, 2*np.pi/50)
    hist_counts_0mm, hist_bins = np.histogram(theta_0mm, bins=bins)
    hist_counts_5mm, hist_bins = np.histogram(theta_5mm, bins=bins)

    plt.figure()
    plt.title(r"Target scattering angle $\theta$ (Ideal compton events)")
    plt.xlabel(r"$theta$ [rad]")
    plt.ylabel("counts")
    plt.hist(theta_0mm, bins=bins, histtype=u"step", color="black", label="0mm", density=True)
    plt.hist(theta_5mm, bins=bins, histtype=u"step", color="red", label="5mm", density=True)
    plt.errorbar(hist_bins[1:] - np.pi/50, hist_counts_0mm / (np.sum(hist_counts_0mm) * 2*np.pi/50),
                 np.sqrt(hist_counts_0mm) / (np.sum(hist_counts_0mm) * 2*np.pi/50), color="black", fmt=".")
    plt.errorbar(hist_bins[1:] - np.pi/50, hist_counts_5mm / (np.sum(hist_counts_5mm) * 2*np.pi/50),
                 np.sqrt(hist_counts_5mm) / (np.sum(hist_counts_5mm) * 2*np.pi/50), color="red", fmt=".")
    plt.vlines(x=np.mean(theta_0mm), ymin=0, ymax=0.015, linestyles="--", color="black",
               label=r"$\mu$ = {:.1f}".format(np.mean(theta_0mm)))
    plt.vlines(x=np.mean(theta_5mm), ymin=0, ymax=0.015, linestyles="--", color="red",
               label=r"$\mu$ = {:.1f}".format(np.mean(theta_5mm)))
    plt.legend()
    plt.grid()
    plt.show()

    # plot energy distribution
    e_energy_0mm = [targets_0mm[i, 1] for i in range(targets_0mm.shape[0]) if targets_0mm[i, 0] != 0]
    e_energy_5mm = [targets_5mm[i, 1] for i in range(targets_5mm.shape[0]) if targets_5mm[i, 0] != 0]

    bins = np.arange(0.0, 15.0, 0.2)
    hist_counts_0mm, hist_bins = np.histogram(e_energy_0mm, bins=bins)
    hist_counts_5mm, hist_bins = np.histogram(e_energy_5mm, bins=bins)

    plt.figure()
    plt.title(r"Target electron energy (Ideal compton events)")
    plt.xlabel("Energy [MeV]")
    plt.ylabel("counts")
    plt.hist(e_energy_0mm, bins=bins, histtype=u"step", color="black", label="0mm", density=True)
    plt.hist(e_energy_5mm, bins=bins, histtype=u"step", color="red", label="5mm", density=True)
    plt.errorbar(hist_bins[1:] - 0.1, hist_counts_0mm / (np.sum(hist_counts_0mm) * 0.2),
                 np.sqrt(hist_counts_0mm) / (np.sum(hist_counts_0mm) * 0.2), color="black", fmt=".")
    plt.errorbar(hist_bins[1:] - 0.1, hist_counts_5mm / (np.sum(hist_counts_5mm) * 0.2),
                 np.sqrt(hist_counts_5mm) / (np.sum(hist_counts_5mm) * 0.2), color="red", fmt=".")
    plt.legend()
    plt.grid()
    plt.show()

    # plot photon distribution
    p_energy_0mm = [targets_0mm[i, 2] for i in range(targets_0mm.shape[0]) if targets_0mm[i, 0] != 0]
    p_energy_5mm = [targets_5mm[i, 2] for i in range(targets_5mm.shape[0]) if targets_5mm[i, 0] != 0]

    bins = np.arange(0.0, 15.0, 0.2)
    hist_counts_0mm, hist_bins = np.histogram(p_energy_0mm, bins=bins)
    hist_counts_5mm, hist_bins = np.histogram(p_energy_5mm, bins=bins)

    plt.figure()
    plt.title(r"Target photon energy (Ideal compton events)")
    plt.xlabel("Energy [MeV]")
    plt.ylabel("counts")
    plt.hist(p_energy_0mm, bins=bins, histtype=u"step", color="black", label="0mm", density=True)
    plt.hist(p_energy_5mm, bins=bins, histtype=u"step", color="red", label="5mm", density=True)
    plt.errorbar(hist_bins[1:] - 0.1, hist_counts_0mm / (np.sum(hist_counts_0mm) * 0.2),
                 np.sqrt(hist_counts_0mm) / (np.sum(hist_counts_0mm) * 0.2), color="black", fmt=".")
    plt.errorbar(hist_bins[1:] - 0.1, hist_counts_5mm / (np.sum(hist_counts_5mm) * 0.2),
                 np.sqrt(hist_counts_5mm) / (np.sum(hist_counts_5mm) * 0.2), color="red", fmt=".")
    plt.legend()
    plt.grid()
    plt.show()



readout_awal_target()


########################################################################################################################
# Define all quantities to be extracted from the root files because one will only iterate ones


def root_stat_readout():
    preprocessing_0mm = Simulation(dir_main + root_files.optimized_0mm_local)
    preprocessing_5mm = Simulation(dir_main + root_files.optimized_5mm_local)
    print("loaded file: ", root_files.optimized_0mm_local)
    print("loaded file: ", root_files.optimized_5mm_local)

    # general statistics 0mm
    n_total_0mm = preprocessing_0mm.num_entries
    n_valid_0mm = 0
    n_compton_0mm = 0
    n_compton_complete_0mm = 0
    n_compton_ideal_0mm = 0

    # general statistics 5mm
    n_total_5mm = preprocessing_5mm.num_entries
    n_valid_5mm = 0
    n_compton_5mm = 0
    n_compton_complete_5mm = 0
    n_compton_ideal_5mm = 0

    for event in preprocessing_0mm.iterate_events():
        if not event.is_distributed_clusters:
            continue
        # count all statistics
        n_valid_0mm += 1 if event.is_distributed_clusters else 0
        n_compton_0mm += 1 if event.is_compton else 0
        n_compton_complete_0mm += 1 if event.is_complete_compton else 0
        n_compton_ideal_0mm += 1 if event.is_ideal_compton else 0

    for event in preprocessing_5mm.iterate_events():
        if not event.is_distributed_clusters:
            continue
        # count all statistics
        n_valid_5mm += 1 if event.is_distributed_clusters else 0
        n_compton_5mm += 1 if event.is_compton else 0
        n_compton_complete_5mm += 1 if event.is_complete_compton else 0
        n_compton_ideal_5mm += 1 if event.is_ideal_compton else 0

    print("\nEvent statistics: ")
    print("Total events: 0mm: {} | 5mm: {}".format(n_total_0mm, n_total_5mm))
    print("Valid events: 0mm: {:.1f}% | 5mm: {:.1f}%".format(n_valid_0mm / n_total_0mm * 100,
                                                             n_valid_5mm / n_total_5mm * 100))
    print("Compton events: 0mm: {:.1f}% | 5mm: {:.1f}%".format(n_compton_0mm / n_total_0mm * 100,
                                                               n_compton_5mm / n_total_5mm * 100))
    print("Complete compton events: 0mm: {:.1f}% | 5mm: {:.1f}%".format(n_compton_complete_0mm / n_total_0mm * 100,
                                                                        n_compton_complete_5mm / n_total_5mm * 100))
    print("Ideal compton events: 0mm: {:.1f}% | 5mm: {:.1f}%".format(n_compton_ideal_0mm / n_total_0mm * 100,
                                                                     n_compton_ideal_5mm / n_total_5mm * 100))


########################################################################################################################

def root_eventtype_readout():
    preprocessing_0mm = Simulation(dir_main + root_files.optimized_0mm_local)
    preprocessing_5mm = Simulation(dir_main + root_files.optimized_5mm_local)
    print("loaded file: ", root_files.optimized_0mm_local)
    print("loaded file: ", root_files.optimized_5mm_local)

    real_e_pos_z_0mm = []
    real_e_pos_z_5mm = []
    rand_e_pos_z_0mm = []
    rand_e_pos_z_5mm = []

    for event in preprocessing_0mm.iterate_events():
        if event.event_type in [2, 3] and event.real_compton_pos.x != 0.0:
            real_e_pos_z_0mm.append(event.real_compton_pos.z)

    for event in preprocessing_5mm.iterate_events():
        if event.event_type in [2, 3] and event.real_compton_pos.x != 0.0:
            real_e_pos_z_5mm.append(event.real_compton_pos.z)

    # plot electron position z distribution
    bins = np.arange(-98.8 / 2 - 1.3 / 2, 98.8 / 2 + 1.3 / 2, 1.3)

    plt.figure()
    plt.title("MC Compton position real coincidence (+ pileup)")
    plt.xlabel("Position [mm]")
    plt.ylabel("counts")
    plt.hist(real_e_pos_z_0mm, bins=bins, histtype=u"step", color="black", label="0mm", density=True)
    plt.hist(real_e_pos_z_5mm, bins=bins, histtype=u"step", color="red", label="5mm", density=True)
    plt.legend()
    plt.grid()
    plt.show()


def root_source_pos_readout():
    preprocessing_0mm = Simulation(dir_main + root_files.optimized_0mm_local)
    preprocessing_5mm = Simulation(dir_main + root_files.optimized_5mm_local)
    print("loaded file: ", root_files.optimized_0mm_local)
    print("loaded file: ", root_files.optimized_5mm_local)

    src_z_0mm = []
    src_z_5mm = []

    for event in preprocessing_0mm.iterate_events():
        if event.is_ideal_compton:
            src_z_0mm.append(event.real_src_pos.z)

    for event in preprocessing_5mm.iterate_events():
        if event.is_ideal_compton:
            src_z_5mm.append(event.real_src_pos.z)

    # plot electron position z distribution
    bins = np.arange(-80, 20, 1.0)
    hist_counts_0mm, hist_bins_0mm = np.histogram(src_z_0mm, bins=bins)
    hist_counts_5mm, hist_bins_5mm = np.histogram(src_z_5mm, bins=bins)

    plt.figure()
    plt.title("MC Source position (ideal compton events)")
    plt.xlabel("z-position [mm]")
    plt.ylabel("counts")
    plt.hist(src_z_0mm, bins=bins, histtype=u"step", color="black", label="0mm", density=True)
    plt.hist(src_z_5mm, bins=bins, histtype=u"step", color="red", label="5mm", density=True)
    plt.errorbar(hist_bins_0mm[1:] - 0.5, hist_counts_0mm / np.sum(hist_counts_0mm),
                 np.sqrt(hist_counts_0mm) / np.sum(hist_counts_0mm), color="black", fmt=".")
    plt.errorbar(hist_bins_5mm[1:] - 0.5, hist_counts_5mm / np.sum(hist_counts_5mm),
                 np.sqrt(hist_counts_5mm) / np.sum(hist_counts_5mm), color="red", fmt=".")
    plt.legend()
    plt.grid()
    plt.show()

########################################################################################################################
