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


def generic_training():
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


def evaluate_model():
    # Define Model name for training
    model_name = "base_100ep_optimized0mm"
    dir_main = os.getcwd()

    # load the training data
    data = DataModel(dir_main + "/data/" + 'optimized_5mm.npz',
                     batch_size=128, validation_percent=0.0, test_percent=1.0,
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

    # load model
    ai.load("base_100ep_optimized0mm")
    ai.compile_model()

    # evaluate model on test data
    ai.type_threshold = .8
    ai.evaluate()
    print("")

    ai.export_predictions_root("base100ep_optimized5mm_pred.root")


def evaluation_model_FT2():
    dir_main = os.getcwd()

    # model name
    model_name = 'model-2b-lsr-FT2'

    shuffle_clusters = False

    # load the training data
    data = DataModel(dir_main + '/data/optimized_0mm_8cl.npz',
                     batch_size=256, validation_percent=0.1, test_percent=0.2)

    # append an extra dimention to the features since we are using convolutional layers
    data.append_dim = True

    # create an AI instance
    ai = AI(data, model_name)

    # define and create the neural network architecture
    ai.create_model(conv_layers=[128, 64], classifier_layers=[32], type_layers=[8],
                    pos_layers=[64, 32], energy_layers=[32, 16], base_l2=.0001, limbs_l2=.0001)

    # LOADING
    ai.load(model_name)
    ai.compile_model()

    # evaluate the AI on the test set
    ai.evaluate()

    ai.export_predictions_root(root_name='model2blsrzft2_optimized0mm_full.root')

def training_finetuning():
    dir_main = os.getcwd()

    # model name
    model_name = 'base_100ep_optimized0mm-FT2'

    # source model name to load the network weights
    source_model = 'base_100ep_optimized0mm'

    shuffle_clusters = False

    # load the training data
    data = DataModel(dir_main + '/data/optimized_0mm_training.npz',
                     batch_size=256, validation_percent=.05, test_percent=.1)

    # append an extra dimention to the features since we are using convolutional layers
    data.append_dim = True

    # create an AI instance
    ai = AI(data, model_name)

    # randomly shuffle the training data
    np.random.seed(888)
    ai.data.shuffle(only_train=False)

    # shuffle the clusters within each event
    if shuffle_clusters:
        ai.data.shuffle_training_clusters()

    # define the priority of selection
    ai.data.weight_non_compton = .2

    # define the learning rate scheduler
    def lr_scheduler(epoch):
        if epoch < (14):
            return .00001
        else:
            return .000003

    # define and create the neural network architecture
    ai.create_model(conv_layers=[128, 64], classifier_layers=[32], type_layers=[16,8],
                    pos_layers=[64, 32], energy_layers=[32, 16], base_l2=.0001, limbs_l2=.0001)

    # load the weights of the source model
    ai.load(source_model, optimizer=False)
    ai.compile_model()

    # locate the correctly reconstructed events and replace the type target
    pred = ai.predict(ai.data.get_features())
    true_type = ai.data._targets[:, 0].copy()
    sp_type = ai._find_matches(ai.data._targets, pred, keep_length=True)
    ai.data._targets[:, 0] = sp_type

    # eliminate the components weight not intended for tuning
    ai.weight_type = .05 * 1
    ai.weight_e_cluster = .15 * 0
    ai.weight_p_cluster = .1 * 0
    ai.weight_pos_x = 12 * 0
    ai.weight_pos_y = 2 * 0
    ai.weight_pos_z = 8 * 0
    ai.weight_energy = 7 * 0

    # freeze all network components
    for layer in ai.model.layers:
        layer.trainable = False

    # defreeze the parts to be tuned
    for layer_name in ['dense_type_1', 'type']:
        layer = ai.model.get_layer(layer_name)
        layer.trainable = True

    # print the trainable layers
    for layer in ai.model.layers:
        if layer.trainable:
            print('{:17s}{}'.format(layer.name, layer.trainable))
    print()

    # compile the AI
    ai.compile_model(learning_rate=.00001)
    l_callbacks = [
        keras.callbacks.LearningRateScheduler(lr_scheduler),
    ]

    # start the training of the network
    ai.train(epochs=20, shuffle_clusters=shuffle_clusters,
             verbose=0, callbacks=l_callbacks)
    print()

    # restore the true event type
    ai.data._targets[:, 0] = true_type

    # evaluate the AI on the test set
    ai.evaluate()

    # save the trained model
    ai.save(file_name=model_name)

evaluate_model()