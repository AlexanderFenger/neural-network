# header
import numpy as np
import os
import matplotlib.pyplot as plt

# load MCTRUTH files (.npz format)
# define file paths
dir_main = os.getcwd()
dir_data = dir_main + "/data/"
dir_plots = dir_main + "/plots/"

# filenames
filename1 = "optimized_0mm_MCTruth.npz"
filename2 = "optimized_5mm_MCTruth.npz"

# load initial npz file
data_0mm = np.load(dir_data + filename1)
data_5mm = np.load(dir_data + filename2)

mc_0mm = data_0mm["MC_TRUTH"]
mc_5mm = data_5mm["MC_TRUTH"]


#######################################################################

def print_classifier_stat(df):
    """
    takes MC-Truth array and print overall classifier statistics
    """
    # print overall statistics
    n_cb_classified = 0
    n_nn_classified = 0  # given by tag 0
    n_nn_classified_list = [0, 0, 0]  # (wrong, position match, correct)
    n_matched = 0
    for i in range(df.shape[0]):
        # position 2 CB classification tag
        if df[i, 2] != 0:
            n_cb_classified += 1
        # position 3 NN classification tag
        if df[i, 3] in [0, 1, 2]:
            n_nn_classified += 1
            n_nn_classified_list[int(df[i, 3])] += 1
        # matched classification
        if (df[i, 3] in [1, 2] and df[i, 2] != 0):
            n_matched += 1

    print("\n### Dataset: ")
    print("Total events: ", mc_0mm.shape[0])
    print("CB classified: ", n_cb_classified)
    print("NN classified: ", n_nn_classified)
    print("wrong: ", n_nn_classified_list[0])
    print("position match: ", n_nn_classified_list[1])
    print("correct: ", n_nn_classified_list[2])
    print("matched: ", n_matched)


# print_classifier_stat(mc_0mm)
# print_classifier_stat(mc_5mm)

###########################################################################################
def plot_dist_event_selection(ary_idx, bins, quantity, x_label, save_plots=False):
    """
    plot the distribution of a given quantity (given by array index)
    plot total/ideal compton, ideal compton/CB, ideal compton/NN, ideal compton/CB/NN
    """

    # define bin-range
    bins = bins
    width = bins[1] - bins[0]  # bins should always have same width

    # grab data
    ary_0mm_total = [mc_0mm[i, ary_idx] for i in range(mc_0mm.shape[0]) if mc_0mm[i, 4] != 0.0]
    ary_5mm_total = [mc_5mm[i, ary_idx] for i in range(mc_5mm.shape[0]) if mc_5mm[i, 4] != 0.0]

    ary_0mm_idealcompton = [mc_0mm[i, ary_idx] for i in range(mc_0mm.shape[0]) if (mc_0mm[i, 24])]
    ary_5mm_idealcompton = [mc_5mm[i, ary_idx] for i in range(mc_5mm.shape[0]) if (mc_5mm[i, 24])]

    # filtered == non compton events are filtered out since their MC-Truths filled with zeros
    ary_0mm_cb_filtered = [mc_0mm[i, ary_idx] for i in range(mc_0mm.shape[0]) if
                           (mc_0mm[i, 2] in [1, 3] and mc_0mm[i, 4] != 0)]
    ary_5mm_cb_filtered = [mc_5mm[i, ary_idx] for i in range(mc_5mm.shape[0]) if
                           (mc_5mm[i, 2] in [1, 3] and mc_5mm[i, 4] != 0)]

    ary_0mm_nn = [mc_0mm[i, ary_idx] for i in range(mc_0mm.shape[0]) if (mc_0mm[i, 3] in [2])]
    ary_5mm_nn = [mc_5mm[i, ary_idx] for i in range(mc_5mm.shape[0]) if (mc_5mm[i, 3] in [2])]

    # create histograms
    hist_0mm_total, _ = np.histogram(ary_0mm_total, bins=bins)
    hist_5mm_total, _ = np.histogram(ary_5mm_total, bins=bins)
    hist_0mm_idealcompton, _ = np.histogram(ary_0mm_idealcompton, bins=bins)
    hist_5mm_idealcompton, _ = np.histogram(ary_5mm_idealcompton, bins=bins)
    hist_0mm_cb_filtered, _ = np.histogram(ary_0mm_cb_filtered, bins=bins)
    hist_5mm_cb_filtered, _ = np.histogram(ary_5mm_cb_filtered, bins=bins)
    hist_0mm_nn, _ = np.histogram(ary_0mm_nn, bins=bins)
    hist_5mm_nn, _ = np.histogram(ary_5mm_nn, bins=bins)

    # generate plots
    # MC Total / MC Ideal Compton
    plt.figure()
    plt.title(quantity + " Ideal Compton Events")
    plt.xlabel(x_label)
    plt.ylabel("counts (normalized)")
    # total event histogram
    plt.hist(ary_0mm_total, bins=bins, histtype=u"step", color="black", label="0mm total", density=True, alpha=0.5,
             linestyle="--")
    plt.hist(ary_5mm_total, bins=bins, histtype=u"step", color="red", label="5mm total", density=True, alpha=0.5,
             linestyle="--")
    # ideal compton event histogram
    plt.hist(ary_0mm_idealcompton, bins=bins, histtype=u"step", color="black", label="0mm Ideal Compton",
             density=True)
    plt.hist(ary_5mm_idealcompton, bins=bins, histtype=u"step", color="red", label="5mm Ideal Compton",
             density=True)
    plt.errorbar(bins[1:] - width / 2, hist_0mm_idealcompton / np.sum(hist_0mm_idealcompton) / width,
                 np.sqrt(hist_0mm_idealcompton) / np.sum(hist_0mm_idealcompton) / width, color="black", fmt=".")
    plt.errorbar(bins[1:] - width / 2, hist_5mm_idealcompton / np.sum(hist_5mm_idealcompton) / width,
                 np.sqrt(hist_5mm_idealcompton) / np.sum(hist_5mm_idealcompton) / width, color="red", fmt=".")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save_plots:
        plt.savefig(dir_plots + "eventselection_" + quantity + "_total.png")
    else:
        plt.show()

    # MC Ideal Compton / MC CB
    plt.figure()
    plt.title(quantity + " CB Event Selection")
    plt.xlabel(x_label)
    plt.ylabel("counts (normalized)")
    # total event histogram
    plt.hist(ary_0mm_idealcompton, bins=bins, histtype=u"step", color="black", label="0mm Ideal Compton", density=True,
             alpha=0.5, linestyle="--")
    plt.hist(ary_5mm_idealcompton, bins=bins, histtype=u"step", color="red", label="5mm Ideal Compton", density=True,
             alpha=0.5, linestyle="--")
    # ideal compton event histogram
    plt.hist(ary_0mm_cb_filtered, bins=bins, histtype=u"step", color="black", label="0mm CB correct*",
             density=True)
    plt.hist(ary_5mm_cb_filtered, bins=bins, histtype=u"step", color="red", label="5mm CB correct*",
             density=True)
    plt.errorbar(bins[1:] - width / 2, hist_0mm_cb_filtered / np.sum(hist_0mm_cb_filtered) / width,
                 np.sqrt(hist_0mm_cb_filtered) / np.sum(hist_0mm_cb_filtered) / width, color="black", fmt=".")
    plt.errorbar(bins[1:] - width / 2, hist_5mm_cb_filtered / np.sum(hist_5mm_cb_filtered) / width,
                 np.sqrt(hist_5mm_cb_filtered) / np.sum(hist_5mm_cb_filtered) / width, color="red", fmt=".")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save_plots:
        plt.savefig(dir_plots + "eventselection_" + quantity + "_cb.png")
    else:
        plt.show()

    # MC Ideal Compton / MC NN
    plt.figure()
    plt.title(quantity + " NN Event Selection")
    plt.xlabel(x_label)
    plt.ylabel("counts (normalized)")
    # total event histogram
    plt.hist(ary_0mm_idealcompton, bins=bins, histtype=u"step", color="black", label="0mm Ideal Compton", density=True,
             alpha=0.5, linestyle="--")
    plt.hist(ary_5mm_idealcompton, bins=bins, histtype=u"step", color="red", label="5mm Ideal Compton", density=True,
             alpha=0.5, linestyle="--")
    # ideal compton event histogram
    plt.hist(ary_0mm_nn, bins=bins, histtype=u"step", color="black", label="0mm NN correct",
             density=True)
    plt.hist(ary_5mm_nn, bins=bins, histtype=u"step", color="red", label="5mm NN correct",
             density=True)
    plt.errorbar(bins[1:] - width / 2, hist_0mm_nn / np.sum(hist_0mm_nn) / width,
                 np.sqrt(hist_0mm_nn) / np.sum(hist_0mm_nn) / width, color="black", fmt=".")
    plt.errorbar(bins[1:] - width / 2, hist_5mm_nn / np.sum(hist_5mm_nn) / width,
                 np.sqrt(hist_5mm_nn) / np.sum(hist_5mm_nn) / width, color="red", fmt=".")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save_plots:
        plt.savefig(dir_plots + "eventselection_" + quantity + "_nn.png")
    else:
        plt.show()


plot_dist_event_selection(11, np.arange(-1, 1.01, 0.01), "MCDirection_source_z", "position [mm]", save_plots=True)
