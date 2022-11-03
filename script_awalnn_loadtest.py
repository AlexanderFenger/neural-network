import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

# declare directory path
dir_main = os.getcwd()
dir_data = dir_main + "/data/"

# load training data
npz_data = np.load(dir_data + "optimized_0mm_training.npz")
# features are already flatten
npz_features = npz_data["features"]

# rescale input
for i in range(npz_features.shape[1]):
    npz_features[:, i] = (npz_features[:, i] - np.mean(npz_features[:, i])) / np.std(npz_features[:, i])

# grab training targets
npz_targets = npz_data["targets"]
npz_targets = npz_targets[:, 0]

# define class weights
class_weights = {0: 1/(len(npz_targets) - np.sum(npz_targets)), 1: 1/np.sum(npz_targets)}

# split samples into training and validation pool
x_train, x_valid, y_train, y_valid = train_test_split(npz_features, npz_targets, test_size=0.1, random_state=42)

# model settings
nodes = 128
layers = 3
dropout = 0.2
initializer = tf.keras.initializers.random_normal()
batch_size = 128
epochs = 50

# build simple neural network model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(nodes, input_dim=x_train.shape[1], activation="relu", kernel_initializer=initializer))
for i in range(layers - 1):
    model.add(tf.keras.layers.Dense(nodes, activation="relu", kernel_initializer=initializer))
    model.add(tf.keras.layers.Dropout(dropout / (i + 1)))
model.add(tf.keras.layers.Dense(1, activation="softmax", kernel_initializer=initializer))
model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=["accuracy"])
history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), verbose=1, epochs=epochs,
                    batch_size=batch_size, class_weight=class_weights)
