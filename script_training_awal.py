import numpy as np
import uproot
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split

from sificc_lib_awal.AI import AI
from sificc_lib_awal.utils import utils
from sificc_lib_awal.Event import Event
from sificc_lib_awal.Simulation import Simulation
from sificc_lib.root_files import root_files
from sificc_lib_awal.DataModel import DataModel

"""
# calculate normalization
dir_main = os.getcwd()
simulation = Simulation(dir_main + root_files.optimized_0mm_local)
utils.calculate_normalizations(simulation)
"""

# Define Model name for training
model_name = "base_100ep_optimized0mm"
dir_main = os.getcwd()

# load the training data
data = DataModel(dir_main + "/data/" + 'optimized_0mm_training.npz',
                 batch_size=128, validation_percent=.1, test_percent=.2,
                 weight_compton=1, weight_non_compton=1)

# append an extra dimension to the features since we are using convolutional layers
# 0 padding of convolutional layer?
data.append_dim = True

# create an AI instance
ai = AI(data, model_name)

ai.weight_type = 1.4
ai.weight_pos_x = 2.5
ai.weight_pos_y = .5
ai.weight_pos_z = 2
ai.weight_energy = .8
ai.weight_e_cluster = .6
ai.weight_p_cluster = .4

# define and create the neural network architecture
ai.create_model(conv_layers=[128, 64], classifier_layers=[32], type_layers=[16, 8],
                pos_layers=[64, 32], energy_layers=[32, 16], base_l2=.0000, limbs_l2=.0000)

# compile the ai
ai.compile_model(learning_rate=0.003)


# define the learning rate scheduler for the training phase
# TODO: WEG DAMIT
def lr_scheduler(epoch):
    if epoch < 60:
        return .003
    elif epoch < 110:
        return .001
    elif epoch < 140:
        return .0003
    elif epoch < 165:
        return .0001
    elif epoch < 185:
        return .00003
    elif epoch < 195:
        return .00001
    else:
        return .000003


l_callbacks = [
    keras.callbacks.LearningRateScheduler(lr_scheduler),
]

# balance the training data since there are too many background events
ai.data.balance_training = True

# start the training
ai.train(epochs=100, shuffle_clusters=False, verbose=2, callbacks=l_callbacks)

# evaluate the AI on the training set
ai.model.evaluate(ai.data.train_x, ai.data.train_y, verbose=1)
print()

# plot the training loss
ai.plot_training_loss(smooth=False)

# save the trained model
ai.save(file_name=model_name)
