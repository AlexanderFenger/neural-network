import numpy as np
import matplotlib.pyplot as plt
import os

###############################################

dir_main = os.getcwd()
filename = "data_test.npz"

# grab training data from npz file
data = np.load(dir_main + "/data/" + filename)
data_features = data["features"]


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


plot_cluster_perc(data_features)
