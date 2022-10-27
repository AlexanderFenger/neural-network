# test script for all crucial libraries

# test base packages and print versions
import numpy
import tensorflow
from tensorflow import keras

import uproot
print("uproot: version ", uproot.__version__)
import uproot_methods
print("uproot_methods: version ", uproot_methods.__version__)
import pandas
import tqdm
import datetime

# needed in future
from sklearn.model_selection import train_test_split

# TODO: test for sificcnn library

# test tensorflow GPU availability
print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))
print(tensorflow.config.list_physical_devices('GPU'))