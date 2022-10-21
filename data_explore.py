import numpy as np
import os
import matplotlib.pyplot as plt
# test for extracting event information from root files
from sificc_lib_awal import Simulation
from sificc_lib_awal import utils
from sificc_lib.root_files import root_files

########################################################################################################################

dir_main = os.getcwd()

# test print to check functionality
preprocessing_0mm = Simulation(dir_main + root_files.optimized_0mm_local)
preprocessing_5mm = Simulation(dir_main + root_files.optimized_5mm_local)
print("loaded file: ", root_files.optimized_0mm_local)
print("loaded file: ", root_files.optimized_5mm_local)

#utils.show_simulation_setup(preprocessing_0mm)


########################################################################################################################
# Define all quantities to be extracted from the root files because one will only iterate ones

def root_stat_readout():
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

########################################################################################################################
