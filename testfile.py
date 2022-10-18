import numpy as np
import os


########################################################################################################################

def test_print():
    # test for extracting event information from root files
    from sificc_lib.Preprocessing import Preprocessing
    from sificc_lib.utilities import utilities
    from sificc_lib.root_files import root_files
    from sificc_lib.Plotter import Plotter

    preprocessing = Preprocessing(root_files.optimized_0mm)
    print("loaded file: ", root_files.optimized_0mm)
    utilities.print_event_summary(preprocessing)

def export_npz():
    # extract Awals features as csv file
    dir_main = os.getcwd()

    from sificc_lib_awal.Simulation import Simulation
    from sificc_lib_awal.Event import Event
    from sificc_lib.root_files import root_files
    from sificc_lib_awal.DataModel import DataModel
    import csv

    simulation = Simulation(root_files.optimized_0mm)
    DataModel.generate_training_data(simulation=simulation, output_name=dir_main + "/data/" + 'data_test.npz')

print("Start testing")
export_npz()

