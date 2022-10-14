import numpy as np

from sificc_lib.Preprocessing import Preprocessing
from sificc_lib.utilities import utilities
from sificc_lib.root_files import root_files

preprocessing = Preprocessing(root_files.root_test)
print("loaded file: ", root_files.root_test)
utilities.print_event_summary(preprocessing)
