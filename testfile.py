import numpy as np
import os


########################################################################################################################

"""
# test for extracting event information from root files
from sificc_lib.Preprocessing import Preprocessing
from sificc_lib.utilities import utilities
from sificc_lib.root_files import root_files
from sificc_lib.Plotter import Plotter

preprocessing = Preprocessing(root_files.root_test)
print("loaded file: ", root_files.root_test)
utilities.print_event_summary(preprocessing)
"""

# extract Awals features as csv file
dir_main = os.getcwd()

from sificc_lib_awal.Simulation import Simulation
from sificc_lib_awal.Event import Event
from sificc_lib.root_files import root_files
import csv

simulation = Simulation(root_files.root_test)



