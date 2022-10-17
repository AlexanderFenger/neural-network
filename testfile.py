import numpy as np
import os

from sificc_lib.Preprocessing import Preprocessing
from sificc_lib.utilities import utilities
from sificc_lib.root_files import root_files
from sificc_lib.Plotter import Plotter

preprocessing = Preprocessing(root_files.root_test)
print("loaded file: ", root_files.root_test)
utilities.print_event_summary(preprocessing)

dir_main = os.getcwd()
plotter = Plotter(dir_main + "/plots/")
plotter.plot_source_dist(preprocessing, "hist2d_test")