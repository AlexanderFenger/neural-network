import numpy as np
import os
import uproot
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow.keras.backend as K

# declare directory path
dir_main = os.getcwd()
dir_data = dir_main + "/data/"

# load training data
npz_train = np.load(dir_data + "optimized_0mm_training.npz")
x_train = npz_train["features"]
y_train = npz_train["targets"]

# create Neural network
