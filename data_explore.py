import numpy as np
import os

# test for extracting event information from root files
from sificc_lib.Preprocessing import Preprocessing
from sificc_lib.utilities import utilities
from sificc_lib.root_files import root_files

########################################################################################################################

# test print to check functionality
preprocessing = Preprocessing(root_files.optimized_0mm)
print("loaded file: ", root_files.optimized_0mm)
utilities.print_event_summary(preprocessing)

# print simulation setup
print("\nLoad simulation setup: ")
print('Scatterer:')
print('\tPosition: ({:.1f}, {:.1f}, {:.1f})'.format(preprocessing.scatterer.position.x,
                                                    preprocessing.scatterer.position.y,
                                                    preprocessing.scatterer.position.z))
print('\tThickness: ({:.1f}, {:.1f}, {:.1f})'.format(preprocessing.scatterer.thickness_x,
                                                     preprocessing.scatterer.thickness_y,
                                                     preprocessing.scatterer.thickness_z))
print('\nAbsorber:')
print('\tPosition: ({:.1f}, {:.1f}, {:.1f})'.format(preprocessing.scatterer.position.x,
                                                    preprocessing.absorber.position.y,
                                                    preprocessing.absorber.position.z))
print('\tThickness: ({:.1f}, {:.1f}, {:.1f})'.format(preprocessing.absorber.thickness_x,
                                                     preprocessing.absorber.thickness_y,
                                                     preprocessing.absorber.thickness_z))

# print general event statistics
n_total = preprocessing.num_entries
n_valid = 0
n_compton = 0
n_compton_complete = 0
n_compton_ideal = 0

for event in preprocessing.iterate_events():
    if not event.is_valid:
        continue
    # count all statistics
    n_valid += 1 if event.is_valid else 0
    n_compton += 1 if event.is_compton else 0
    n_compton_complete += 1 if event.is_compton_complete else 0
    n_compton_ideal += 1 if event.is_compton_ideal else 0

print("\nEvent statistics: ")
print("Total events: ", n_total)
print("Valid events: ", n_valid)
print("Compton events: ", n_compton)
print("Complete compton events: ", n_compton_complete)
print("Ideal compton events: ", n_compton_ideal)
