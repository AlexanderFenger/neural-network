import numpy as np
import matplotlib.pyplot as plt
import os

###############################################

dir_main = os.getcwd()
filename = "data_test.npz"


# grab training data from npz file
# data = np.load(dir_main + "/data/" + filename)
# data_features = data["features"]
# data_targets = data["targets"]
# data_reco = data["reco"]
# data_sequence = data["sequence"]

def sort_event(ary, max_clusters=6):
    ary_new = np.zeros((max_clusters, 9))
    for i in range(max_clusters):
        ary_new[i, :] = ary[i * 9: i * 9 + 9]
    return ary_new


def plot_cluster_perc(data):
    """plot the percentage of clusters being filled for every cluster index"""
    print("plotting percentage of filled clusters:")
    max_cluster = 6

    results = np.zeros((max_cluster))

    for i in range(data.shape[0]):
        # get event from data and convert information to ary
        ary_event = sort_event(data[i, :])

        for j in range(max_cluster):
            if np.sum(ary_event[j, :]) != 0.0:
                results[j] += 1
    results /= data.shape[0]
    results = 1 - results

    # create plot
    plt.figure()
    plt.plot(np.array([1, 2, 3, 4, 5, 6]), results, color="black")
    plt.xlabel("cluster idx")
    plt.ylabel("% of filled clusters")
    plt.grid()
    plt.show()
    plt.savefig(dir_main + "/plots/" + "perc_clusters.png")


def train_test_split(filename, r):
    """take a npz file and split it into training and testing sample"""
    # r: percentage of data towards training set

    # load initial npz file
    print("loading ", dir_main + "/data/" + filename + ".npz")
    data = np.load(dir_main + "/data/" + filename + ".npz")

    data_features = data["features"]
    data_targets = data["targets"]
    data_reco = data["reco"]
    data_sequence = data["sequence"]

    # generate index sequence, shuffle sequence and sample by ratio
    idx = np.linspace(0, data_features.shape[0] - 1, data_features.shape[0], dtype=int)
    np.random.shuffle(idx)
    idx_stop = int(len(idx) * r)
    idx_train = idx[0:idx_stop]
    idx_test = idx[idx_stop + 1:]

    # generate npz files
    print("generating training set")
    with open(dir_main + "/data/" + filename + "_training.npz", 'wb') as f_train:
        np.savez_compressed(f_train,
                            features=data_features[idx_train, :],
                            targets=data_targets[idx_train, :],
                            reco=data_reco[idx_train, :],
                            sequence=data_sequence[idx_train])
    print("generating test set")
    with open(dir_main + "/data/" + filename + "_test.npz", 'wb') as f_train:
        np.savez_compressed(f_train,
                            features=data_features[idx_test, :],
                            targets=data_targets[idx_test, :],
                            reco=data_reco[idx_test, :],
                            sequence=data_sequence[idx_test])


def plot_pos_dist(filename):
    """plot the distribution of the real e position and rel p position"""
    # load initial npz file
    print("loading ", dir_main + "/data/" + filename + ".npz")
    data = np.load(dir_main + "/data/" + filename + ".npz")
    data_targets = data["targets"]

    list_ey = []
    list_ez = []
    list_py = []
    list_pz = []
    for i in range(data_targets.shape[0]):
        if data_targets[i, 0] == 1:
            list_ey.append(data_targets[i, 4])
            list_ez.append(data_targets[i, 5])

            list_py.append(data_targets[i, 7])
            list_pz.append(data_targets[i, 8])

    # plot
    plt.figure()
    plt.hist2d(list_ey, list_ez, bins=20)
    plt.title("ideal compton events, scatterer, MC e position")
    plt.xlabel("y position [mm]")
    plt.ylabel("z position [mm]")
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.hist2d(list_py, list_pz, bins=20)
    plt.title("ideal compton events, absorber, MC e position")
    plt.xlabel("y position [mm]")
    plt.ylabel("z position [mm]")
    plt.colorbar()
    plt.show()


def export_npz():
    # extract Awals features as npz file
    dir_main = os.getcwd()

    from sificc_lib_awal.Simulation import Simulation
    from sificc_lib_awal.Event import Event
    from sificc_lib.root_files import root_files
    from sificc_lib_awal.DataModel import DataModel
    import csv

    simulation = Simulation(dir_main + root_files.optimized_5mm_local)
    DataModel.generate_training_data(simulation=simulation, output_name=dir_main + "/data/" + 'optimized_5mm.npz')


train_test_split("optimized_5mm", 0.9)
