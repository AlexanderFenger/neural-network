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
    print("Total events: ", df.shape[0])
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

    # events where both classifier agree

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


#######################################################################################################################

def plot_dist_stacked(ary_idx, bins, quantity, x_label, correct=False, save_plots=False):
    """
    plot the distribution of a given quantity (given by array index)
    plot total/ideal compton, ideal compton/CB, ideal compton/NN, ideal compton/CB/NN
    as a stacked histograms
    """

    # define bin-range
    bins = bins
    width = bins[1] - bins[0]  # bins should always have same width

    if correct:
        cb_con = [1, 3]
        nn_con = [2]
    else:
        cb_con = [-1, -3, 1, 3]
        nn_con = [0, 1, 2]

    # grab data depending on event selection:
    # Total: all ideal compton events
    # Non-selected, NN non-CB selected, selected, CB non-NN selected, NN/CB selected
    # all selected events are based on their
    ary_0mm_idealcompton = [mc_0mm[i, ary_idx] for i in range(mc_0mm.shape[0]) if (mc_0mm[i, 24])]
    ary_5mm_idealcompton = [mc_5mm[i, ary_idx] for i in range(mc_5mm.shape[0]) if (mc_5mm[i, 24])]
    # CB-selected but not NN-selected
    ary_0mm_cb = [mc_0mm[i, ary_idx] for i in range(mc_0mm.shape[0]) if
                  (mc_0mm[i, 24] and mc_0mm[i, 2] in cb_con and mc_0mm[i, 3] not in nn_con)]
    ary_5mm_cb = [mc_5mm[i, ary_idx] for i in range(mc_5mm.shape[0]) if
                  (mc_5mm[i, 24] and mc_5mm[i, 2] in cb_con and mc_5mm[i, 3] not in nn_con)]
    # NN-selected but not CB-selected
    ary_0mm_nn = [mc_0mm[i, ary_idx] for i in range(mc_0mm.shape[0]) if
                  (mc_0mm[i, 24] and mc_0mm[i, 2] not in cb_con and mc_0mm[i, 3] in nn_con)]
    ary_5mm_nn = [mc_5mm[i, ary_idx] for i in range(mc_5mm.shape[0]) if
                  (mc_5mm[i, 24] and mc_5mm[i, 2] not in cb_con and mc_5mm[i, 3] in nn_con)]
    # Non-selected
    ary_0mm_non = [mc_0mm[i, ary_idx] for i in range(mc_0mm.shape[0]) if
                   (mc_0mm[i, 24] and mc_0mm[i, 2] not in cb_con and mc_0mm[i, 3] not in nn_con)]
    ary_5mm_non = [mc_5mm[i, ary_idx] for i in range(mc_5mm.shape[0]) if
                   (mc_5mm[i, 24] and mc_5mm[i, 2] not in cb_con and mc_5mm[i, 3] not in nn_con)]
    # CB/NN-selected
    ary_0mm_both = [mc_0mm[i, ary_idx] for i in range(mc_0mm.shape[0]) if
                    (mc_0mm[i, 24] and mc_0mm[i, 2] in cb_con and mc_0mm[i, 3] in nn_con)]
    ary_5mm_both = [mc_5mm[i, ary_idx] for i in range(mc_5mm.shape[0]) if
                    (mc_5mm[i, 24] and mc_5mm[i, 2] in cb_con and mc_5mm[i, 3] in nn_con)]

    # create histograms
    hist_0mm_idealcompton, _ = np.histogram(ary_0mm_idealcompton, bins=bins)
    hist_5mm_idealcompton, _ = np.histogram(ary_5mm_idealcompton, bins=bins)
    hist_0mm_cb, _ = np.histogram(ary_0mm_cb, bins=bins)
    hist_5mm_cb, _ = np.histogram(ary_5mm_cb, bins=bins)
    hist_0mm_nn, _ = np.histogram(ary_0mm_nn, bins=bins)
    hist_5mm_nn, _ = np.histogram(ary_5mm_nn, bins=bins)
    hist_0mm_non, _ = np.histogram(ary_0mm_non, bins=bins)
    hist_5mm_non, _ = np.histogram(ary_5mm_non, bins=bins)
    hist_0mm_both, _ = np.histogram(ary_0mm_both, bins=bins)
    hist_5mm_both, _ = np.histogram(ary_5mm_both, bins=bins)

    # Define x-limits of stacked histograms
    # either the second last bin entry or the first bin with zero entries (due to normalization)
    # yes you can use np.where but that function is dumb
    idx_0mm = len(bins) - 2
    idx_5mm = len(bins) - 2
    for i in range(len(hist_0mm_idealcompton)):
        if hist_0mm_idealcompton[i] == 0:
            idx_0mm = int(i)
            break
    for j in range(len(hist_5mm_idealcompton)):
        if hist_5mm_idealcompton[j] == 0:
            idx_5mm = int(j)
            break

    #######################################################
    # new plot

    # normalized stacked bar plot
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    axs[0].set_title(quantity + " stacked histogram (0mm)")
    axs[0].set_ylabel("counts")
    axs[0].set_xlim(bins[1], bins[idx_0mm + 1])
    axs[0].errorbar(bins[1:] - width / 2, hist_0mm_idealcompton, np.sqrt(hist_0mm_idealcompton),
                    color="black", fmt=".", label="Ideal Compton")
    axs[0].bar(bins[1:] - width / 2, hist_0mm_non,
               width=width, align="center", color="tab:green", label="None")
    axs[0].bar(bins[1:] - width / 2, hist_0mm_cb, bottom=hist_0mm_non,
               width=width, align="center", color="tab:blue", label="CB-only")
    axs[0].bar(bins[1:] - width / 2, hist_0mm_nn, bottom=hist_0mm_non + hist_0mm_cb,
               width=width, align="center", color="tab:red", label="NN-only")
    axs[0].bar(bins[1:] - width / 2, hist_0mm_both, bottom=hist_0mm_non + hist_0mm_cb + hist_0mm_nn,
               width=width, align="center", color="tab:olive", label="CB/NN")
    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel("counts (bin-normalized)")
    axs[1].set_xlim(bins[1], bins[idx_0mm + 1])
    axs[1].set_ylim(0.0, 1.1)
    axs[1].errorbar(bins[1:idx_0mm + 1] - width / 2, hist_0mm_idealcompton[:idx_0mm] / hist_0mm_idealcompton[:idx_0mm],
                    np.sqrt(hist_0mm_idealcompton[:idx_0mm]) / hist_0mm_idealcompton[:idx_0mm],
                    color="black", fmt=".", label="Ideal Compton")
    axs[1].bar(bins[1:idx_0mm + 1] - width / 2, hist_0mm_non[:idx_0mm] / hist_0mm_idealcompton[:idx_0mm],
               width=width, align="center", color="tab:green", label="None")
    axs[1].bar(bins[1:idx_0mm + 1] - width / 2, hist_0mm_cb[:idx_0mm] / hist_0mm_idealcompton[:idx_0mm],
               bottom=hist_0mm_non[:idx_0mm] / hist_0mm_idealcompton[:idx_0mm],
               width=width, align="center", color="tab:blue", label="CB")
    axs[1].bar(bins[1:idx_0mm + 1] - width / 2, hist_0mm_nn[:idx_0mm] / hist_0mm_idealcompton[:idx_0mm],
               bottom=(hist_0mm_non[:idx_0mm] / hist_0mm_idealcompton[:idx_0mm] +
                       hist_0mm_cb[:idx_0mm] / hist_0mm_idealcompton[:idx_0mm]),
               width=width, align="center", color="tab:red", label="NN")
    axs[1].bar(bins[1:idx_0mm + 1] - width / 2, hist_0mm_both[:idx_0mm] / hist_0mm_idealcompton[:idx_0mm],
               bottom=(hist_0mm_non[:idx_0mm] / hist_0mm_idealcompton[:idx_0mm] +
                       hist_0mm_cb[:idx_0mm] / hist_0mm_idealcompton[:idx_0mm] +
                       hist_0mm_nn[:idx_0mm] / hist_0mm_idealcompton[:idx_0mm]),
               width=width, align="center", color="tab:olive", label="CB/NN")
    fig.tight_layout()
    axs[0].legend(loc="upper right")
    if save_plots:
        plt.savefig(dir_plots + "histstacked_" + quantity + "_0mm.png")
    else:
        plt.show()

    # normalized stacked bar plot
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    axs[0].set_title(quantity + " stacked histogram (5mm)")
    axs[0].set_ylabel("counts")
    axs[0].set_xlim(bins[1], bins[idx_5mm + 1])
    axs[0].errorbar(bins[1:] - width / 2, hist_5mm_idealcompton, np.sqrt(hist_5mm_idealcompton),
                    color="black", fmt=".", label="Ideal Compton")
    axs[0].bar(bins[1:] - width / 2, hist_5mm_non,
               width=width, align="center", color="tab:green", label="None")
    axs[0].bar(bins[1:] - width / 2, hist_5mm_cb, bottom=hist_5mm_non,
               width=width, align="center", color="tab:blue", label="CB-only")
    axs[0].bar(bins[1:] - width / 2, hist_5mm_nn, bottom=hist_5mm_non + hist_5mm_cb,
               width=width, align="center", color="tab:red", label="NN-only")
    axs[0].bar(bins[1:] - width / 2, hist_5mm_both, bottom=hist_5mm_non + hist_5mm_cb + hist_5mm_nn,
               width=width, align="center", color="tab:olive", label="CB/NN")
    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel("counts (bin-normalized)")
    axs[1].set_xlim(bins[1], bins[idx_5mm + 1])
    axs[1].set_ylim(0.0, 1.1)
    axs[1].errorbar(bins[1:idx_5mm + 1] - width / 2, hist_5mm_idealcompton[:idx_5mm] / hist_5mm_idealcompton[:idx_5mm],
                    np.sqrt(hist_5mm_idealcompton[:idx_5mm]) / hist_5mm_idealcompton[:idx_5mm],
                    color="black", fmt=".", label="Ideal Compton")
    axs[1].bar(bins[1:idx_5mm + 1] - width / 2, hist_5mm_non[:idx_5mm] / hist_5mm_idealcompton[:idx_5mm],
               width=width, align="center", color="tab:green", label="None")
    axs[1].bar(bins[1:idx_5mm + 1] - width / 2, hist_5mm_cb[:idx_5mm] / hist_5mm_idealcompton[:idx_5mm],
               bottom=hist_5mm_non[:idx_5mm] / hist_5mm_idealcompton[:idx_5mm],
               width=width, align="center", color="tab:blue", label="CB")
    axs[1].bar(bins[1:idx_5mm + 1] - width / 2, hist_5mm_nn[:idx_5mm] / hist_5mm_idealcompton[:idx_5mm],
               bottom=(hist_5mm_non[:idx_5mm] / hist_5mm_idealcompton[:idx_5mm] +
                       hist_5mm_cb[:idx_5mm] / hist_5mm_idealcompton[:idx_5mm]),
               width=width, align="center", color="tab:red", label="NN")
    axs[1].bar(bins[1:idx_5mm + 1] - width / 2, hist_5mm_both[:idx_5mm] / hist_5mm_idealcompton[:idx_5mm],
               bottom=(hist_5mm_non[:idx_5mm] / hist_5mm_idealcompton[:idx_5mm] +
                       hist_5mm_cb[:idx_5mm] / hist_5mm_idealcompton[:idx_5mm] +
                       hist_5mm_nn[:idx_5mm] / hist_5mm_idealcompton[:idx_5mm]),
               width=width, align="center", color="tab:olive", label="CB/NN")
    fig.tight_layout()
    axs[0].legend(loc="upper right")
    if save_plots:
        plt.savefig(dir_plots + "histstacked_" + quantity + "_5mm.png")
    else:
        plt.show()


#######################################################################################################################

def plot_scatter_energy(ary_idx1, ary_idx2, quantity1, quantity2, x_label, y_label, correct=False, save_plots=False):
    """
    plot 4 scatter plots for 2 quantities given by ary_idx1 and ary_idx2
    scatter plot are based on event selection:
    None, CB-only, NN-only, CB/NN
    """
    from matplotlib.colors import LogNorm
    bins = np.arange(0.0, 20.0, 0.5)

    if correct:
        cb_con = [1, 3]
        nn_con = [2]
    else:
        cb_con = [-1, -3, 1, 3]
        nn_con = [0, 1, 2]

    # grab data depending on event selection:
    # Total: all ideal compton events
    # Non-selected, NN non-CB selected, selected, CB non-NN selected, NN/CB selected
    # selection of quantity 1
    # CB-selected but not NN-selected
    ary1_0mm_cb = [mc_0mm[i, ary_idx1] for i in range(mc_0mm.shape[0]) if
                   (mc_0mm[i, 24] and mc_0mm[i, 2] in cb_con and mc_0mm[i, 3] not in nn_con)]
    ary1_5mm_cb = [mc_5mm[i, ary_idx1] for i in range(mc_5mm.shape[0]) if
                   (mc_5mm[i, 24] and mc_5mm[i, 2] in cb_con and mc_5mm[i, 3] not in nn_con)]
    # NN-selected but not CB-selected
    ary1_0mm_nn = [mc_0mm[i, ary_idx1] for i in range(mc_0mm.shape[0]) if
                   (mc_0mm[i, 24] and mc_0mm[i, 2] not in cb_con and mc_0mm[i, 3] in nn_con)]
    ary1_5mm_nn = [mc_5mm[i, ary_idx1] for i in range(mc_5mm.shape[0]) if
                   (mc_5mm[i, 24] and mc_5mm[i, 2] not in cb_con and mc_5mm[i, 3] in nn_con)]
    # Non-selected
    ary1_0mm_non = [mc_0mm[i, ary_idx1] for i in range(mc_0mm.shape[0]) if
                    (mc_0mm[i, 24] and mc_0mm[i, 2] not in cb_con and mc_0mm[i, 3] not in nn_con)]
    ary1_5mm_non = [mc_5mm[i, ary_idx1] for i in range(mc_5mm.shape[0]) if
                    (mc_5mm[i, 24] and mc_5mm[i, 2] not in cb_con and mc_5mm[i, 3] not in nn_con)]
    # CB/NN-selected
    ary1_0mm_both = [mc_0mm[i, ary_idx1] for i in range(mc_0mm.shape[0]) if
                     (mc_0mm[i, 24] and mc_0mm[i, 2] in cb_con and mc_0mm[i, 3] in nn_con)]
    ary1_5mm_both = [mc_5mm[i, ary_idx1] for i in range(mc_5mm.shape[0]) if
                     (mc_5mm[i, 24] and mc_5mm[i, 2] in cb_con and mc_5mm[i, 3] in nn_con)]

    # selection of quantity 2
    ary2_0mm_cb = [mc_0mm[i, ary_idx2] for i in range(mc_0mm.shape[0]) if
                   (mc_0mm[i, 24] and mc_0mm[i, 2] in cb_con and mc_0mm[i, 3] not in nn_con)]
    ary2_5mm_cb = [mc_5mm[i, ary_idx2] for i in range(mc_5mm.shape[0]) if
                   (mc_5mm[i, 24] and mc_5mm[i, 2] in cb_con and mc_5mm[i, 3] not in nn_con)]
    # NN-selected but not CB-selected
    ary2_0mm_nn = [mc_0mm[i, ary_idx2] for i in range(mc_0mm.shape[0]) if
                   (mc_0mm[i, 24] and mc_0mm[i, 2] not in cb_con and mc_0mm[i, 3] in nn_con)]
    ary2_5mm_nn = [mc_5mm[i, ary_idx2] for i in range(mc_5mm.shape[0]) if
                   (mc_5mm[i, 24] and mc_5mm[i, 2] not in cb_con and mc_5mm[i, 3] in nn_con)]
    # Non-selected
    ary2_0mm_non = [mc_0mm[i, ary_idx2] for i in range(mc_0mm.shape[0]) if
                    (mc_0mm[i, 24] and mc_0mm[i, 2] not in cb_con and mc_0mm[i, 3] not in nn_con)]
    ary2_5mm_non = [mc_5mm[i, ary_idx2] for i in range(mc_5mm.shape[0]) if
                    (mc_5mm[i, 24] and mc_5mm[i, 2] not in cb_con and mc_5mm[i, 3] not in nn_con)]
    # CB/NN-selected
    ary2_0mm_both = [mc_0mm[i, ary_idx2] for i in range(mc_0mm.shape[0]) if
                     (mc_0mm[i, 24] and mc_0mm[i, 2] in cb_con and mc_0mm[i, 3] in nn_con)]
    ary2_5mm_both = [mc_5mm[i, ary_idx2] for i in range(mc_5mm.shape[0]) if
                     (mc_5mm[i, 24] and mc_5mm[i, 2] in cb_con and mc_5mm[i, 3] in nn_con)]

    # create 0mm scatter plot
    fig, axs = plt.subplots(2, 2)
    plt.suptitle("Scatter plot {}/{} (0mm)".format(quantity1, quantity2))
    axs[0, 0].set_xlim(0, 20)
    axs[0, 0].set_ylim(0, 20)
    h0 = axs[0, 0].hist2d(ary1_0mm_non, ary2_0mm_non, norm=LogNorm(vmin=1, vmax=12000), bins=bins, label="None")
    fig.colorbar(h0[3], ax=axs[0, 0])
    axs[0, 0].text(x=15, y=18, s="None")
    axs[0, 0].set_ylabel(y_label)

    axs[0, 1].set_xlim(0, 20)
    axs[0, 1].set_ylim(0, 20)
    h1 = axs[0, 1].hist2d(ary1_0mm_cb, ary2_0mm_cb, norm=LogNorm(vmin=1, vmax=12000), bins=bins, label="CB-only")
    fig.colorbar(h1[3], ax=axs[0, 1])
    axs[0, 1].text(x=13, y=18, s="CB-Only")

    axs[1, 0].set_xlim(0, 20)
    axs[1, 0].set_ylim(0, 20)
    h2 = axs[1, 0].hist2d(ary1_0mm_nn, ary2_0mm_nn, norm=LogNorm(vmin=1, vmax=12000), bins=bins, label="NN-only")
    fig.colorbar(h2[3], ax=axs[1, 0])
    axs[1, 0].text(x=13, y=18, s="NN-Only")
    axs[1, 0].set_xlabel(x_label)
    axs[1, 0].set_ylabel(y_label)

    axs[1, 1].set_xlim(0, 20)
    axs[1, 1].set_ylim(0, 20)
    h3 = axs[1, 1].hist2d(ary1_0mm_both, ary2_0mm_both, norm=LogNorm(vmin=1, vmax=12000), bins=bins, label="CB/NN")
    fig.colorbar(h3[3], ax=axs[1, 1])
    axs[1, 1].text(x=14, y=18, s="CB/NN")
    axs[1, 1].set_xlabel(x_label)
    if save_plots:
        plt.savefig(dir_plots + "scatterplot_" + quantity1 + quantity2 + "_0mm.png")
    else:
        plt.show()

    # create 5mm scatter plot
    fig, axs = plt.subplots(2, 2)
    plt.suptitle("Scatter plot {}/{} (5mm)".format(quantity1, quantity2))
    axs[0, 0].set_xlim(0, 20)
    axs[0, 0].set_ylim(0, 20)
    h0 = axs[0, 0].hist2d(ary1_5mm_non, ary2_5mm_non, norm=LogNorm(vmin=1, vmax=12000), bins=bins, label="None")
    fig.colorbar(h0[3], ax=axs[0, 0])
    axs[0, 0].text(x=15, y=18, s="None")
    axs[0, 0].set_ylabel(y_label)

    axs[0, 1].set_xlim(0, 20)
    axs[0, 1].set_ylim(0, 20)
    h1 = axs[0, 1].hist2d(ary1_5mm_cb, ary2_5mm_cb, norm=LogNorm(vmin=1, vmax=12000), bins=bins, label="CB-only")
    fig.colorbar(h1[3], ax=axs[0, 1])
    axs[0, 1].text(x=13, y=18, s="CB-Only")

    axs[1, 0].set_xlim(0, 20)
    axs[1, 0].set_ylim(0, 20)
    h2 = axs[1, 0].hist2d(ary1_5mm_nn, ary2_5mm_nn, norm=LogNorm(vmin=1, vmax=12000), bins=bins, label="NN-only")
    fig.colorbar(h2[3], ax=axs[1, 0])
    axs[1, 0].text(x=13, y=18, s="NN-Only")
    axs[1, 0].set_xlabel(x_label)
    axs[1, 0].set_ylabel(y_label)

    axs[1, 1].set_xlim(0, 20)
    axs[1, 1].set_ylim(0, 20)
    h3 = axs[1, 1].hist2d(ary1_5mm_both, ary2_5mm_both, norm=LogNorm(vmin=1, vmax=12000), bins=bins, label="CB/NN")
    fig.colorbar(h3[3], ax=axs[1, 1])
    axs[1, 1].text(x=14, y=18, s="CB/NN")
    axs[1, 1].set_xlabel(x_label)
    if save_plots:
        plt.savefig(dir_plots + "scatterplot_" + quantity1 + quantity2 + "_5mm.png")
    else:
        plt.show()


def plot_scatter_energy_source_pos(correct=False, save_plots=False):
    from matplotlib.colors import LogNorm
    bins1 = np.arange(-30, 10.0, 1.0)
    bins2 = np.arange(0.0, 20.0, 0.5)

    ary_idx1 = 8
    ary_idx2 = 26

    if correct:
        cb_con = [1, 3]
        nn_con = [2]
    else:
        cb_con = [-1, -3, 1, 3]
        nn_con = [0, 1, 2]

    # grab data depending on event selection:
    # Total: all ideal compton events
    # Non-selected, NN non-CB selected, selected, CB non-NN selected, NN/CB selected
    # selection of quantity 1
    # CB-selected but not NN-selected
    ary1_0mm_all = [mc_0mm[i, ary_idx1] for i in range(mc_0mm.shape[0]) if (mc_0mm[i, 24])]
    ary1_5mm_all = [mc_5mm[i, ary_idx1] for i in range(mc_5mm.shape[0]) if (mc_5mm[i, 24])]

    # selection of quantity 2
    ary2_0mm_all = [mc_0mm[i, ary_idx2] for i in range(mc_0mm.shape[0]) if (mc_0mm[i, 24])]
    ary2_5mm_all = [mc_5mm[i, ary_idx2] for i in range(mc_5mm.shape[0]) if (mc_5mm[i, 24])]

    # create scatter plot
    fig = plt.figure()
    plt.title("Scatter plot {}/{} (0mm)".format("MCSource_position.z", "MCEnergyPrimary"))
    plt.xlabel("Position [mm]")
    plt.ylabel("Energy primary [MeV]")
    h0 = plt.hist2d(ary1_0mm_all, ary2_0mm_all, norm=LogNorm(), bins=[bins1, bins2])
    plt.colorbar(h0[3])
    if save_plots:
        plt.savefig(dir_plots + "scatterplot_" + "MCSource_position.z" + "MCEnergyPrimary" + "_0mm.png")
    else:
        plt.show()

    # create scatter plot
    fig = plt.figure()
    plt.title("Scatter plot {}/{} (5mm)".format("MCSource_position.z", "MCEnergyPrimary"))
    plt.xlabel("Position [mm]")
    plt.ylabel("Energy primary [MeV]")
    h0 = plt.hist2d(ary1_5mm_all, ary2_5mm_all, norm=LogNorm(), bins=[bins1, bins2])
    plt.colorbar(h0[3])
    if save_plots:
        plt.savefig(dir_plots + "scatterplot_" + "MCSource_position.z" + "MCEnergyPrimary" + "_5mm.png")
    else:
        plt.show()


def energy_cut(correct=False, save_plots=False, energycut=False):
    import matplotlib.patches as mpatches

    ary_idx = 4
    bins = np.arange(0.0, 10.0, 0.1)
    width = bins[1] - bins[0]  # bins should always have same width

    if correct:
        cb_con = [1, 3]
        nn_con = [2]
    else:
        cb_con = [-1, -3, 1, 3]
        nn_con = [0, 1, 2]

    # energycut
    a = 0.10
    if energycut:
        for i in range(mc_0mm.shape[0]):
            rand = np.random.uniform()
            # energy filter
            if rand > 1 / np.exp(np.sum(mc_0mm[i, 4:6]) * a):
                mc_0mm[i, 2] = 0
            else:
                continue

    if energycut:
        for i in range(mc_5mm.shape[0]):
            rand = np.random.uniform()
            # energy filter
            if rand > 1 / np.exp(np.sum(mc_5mm[i, 4:6]) * a):
                mc_5mm[i, 2] = 0
            else:
                continue

    # all selected events are based on their
    ary_0mm_idealcompton = [np.sum(mc_0mm[i, 4:6]) for i in range(mc_0mm.shape[0]) if (mc_0mm[i, 24])]
    ary_5mm_idealcompton = [np.sum(mc_5mm[i, 4:6]) for i in range(mc_5mm.shape[0]) if (mc_5mm[i, 24])]
    # CB-selected
    ary_0mm_cb = [np.sum(mc_0mm[i, 4:6]) for i in range(mc_0mm.shape[0]) if
                  (mc_0mm[i, 24] and mc_0mm[i, 2] in cb_con)]
    ary_5mm_cb = [np.sum(mc_5mm[i, 4:6]) for i in range(mc_5mm.shape[0]) if
                  (mc_5mm[i, 24] and mc_5mm[i, 2] in cb_con)]
    # NN-selected
    ary_0mm_nn = [np.sum(mc_0mm[i, 4:6]) for i in range(mc_0mm.shape[0]) if
                  (mc_0mm[i, 24] and mc_0mm[i, 3] in nn_con)]
    ary_5mm_nn = [np.sum(mc_5mm[i, 4:6]) for i in range(mc_5mm.shape[0]) if
                  (mc_5mm[i, 24] and mc_5mm[i, 3] in nn_con)]
    # Non-selected
    ary_0mm_non_nn = [np.sum(mc_0mm[i, 4:6]) for i in range(mc_0mm.shape[0]) if
                      (mc_0mm[i, 24] and mc_0mm[i, 3] not in nn_con)]
    ary_5mm_non_nn = [np.sum(mc_5mm[i, 4:6]) for i in range(mc_5mm.shape[0]) if
                      (mc_5mm[i, 24] and mc_5mm[i, 3] not in nn_con)]

    ary_0mm_non_cb = [np.sum(mc_0mm[i, 4:6]) for i in range(mc_0mm.shape[0]) if
                      (mc_0mm[i, 24] and mc_0mm[i, 2] not in cb_con)]
    ary_5mm_non_cb = [np.sum(mc_5mm[i, 4:6]) for i in range(mc_5mm.shape[0]) if
                      (mc_5mm[i, 24] and mc_5mm[i, 2] not in cb_con)]

    event_count_0mm = len([np.sum(mc_0mm[i, 4:6]) for i in range(mc_0mm.shape[0]) if (mc_0mm[i, 2]) in cb_con])
    event_count_5mm = len([np.sum(mc_5mm[i, 4:6]) for i in range(mc_5mm.shape[0]) if (mc_5mm[i, 2]) in cb_con])
    print("event_count 0mm: ", event_count_0mm)
    print("event_count 5mm: ", event_count_5mm)
    print(len(ary_0mm_non_cb), len(ary_0mm_cb), len(ary_0mm_non_cb) + len(ary_0mm_cb))

    # create histograms
    hist_0mm_idealcompton, _ = np.histogram(ary_0mm_idealcompton, bins=bins)
    hist_5mm_idealcompton, _ = np.histogram(ary_5mm_idealcompton, bins=bins)
    hist_0mm_cb, _ = np.histogram(ary_0mm_cb, bins=bins)
    hist_5mm_cb, _ = np.histogram(ary_5mm_cb, bins=bins)
    hist_0mm_nn, _ = np.histogram(ary_0mm_nn, bins=bins)
    hist_5mm_nn, _ = np.histogram(ary_5mm_nn, bins=bins)
    hist_0mm_non_nn, _ = np.histogram(ary_0mm_non_nn, bins=bins)
    hist_5mm_non_nn, _ = np.histogram(ary_5mm_non_nn, bins=bins)
    hist_0mm_non_cb, _ = np.histogram(ary_0mm_non_cb, bins=bins)
    hist_5mm_non_cb, _ = np.histogram(ary_5mm_non_cb, bins=bins)

    # Define x-limits of stacked histograms
    # either the second last bin entry or the first bin with zero entries (due to normalization)
    # yes you can use np.where but that function is dumb
    idx_0mm = 0
    idx_5mm = 0
    for i in range(len(hist_0mm_idealcompton)):
        if hist_0mm_idealcompton[i] == 0:
            idx_0mm = int(i)
            break
    for j in range(len(hist_5mm_idealcompton)):
        if hist_5mm_idealcompton[j] == 0:
            idx_5mm = int(j)
            break

    #######################################################
    # new plot

    # normalized stacked bar plot
    fig, axs = plt.subplots(4, 1, gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    axs[0].set_title("MCEnergy_primary stacked histogram (0mm)")
    axs[0].set_ylabel("counts")
    axs[0].set_xlim(bins[idx_0mm], bins[-1])
    axs[0].errorbar(bins[idx_0mm + 2:] - width / 2, hist_0mm_idealcompton[idx_0mm + 1:],
                    np.sqrt(hist_0mm_idealcompton[idx_0mm + 1:]),
                    color="black", fmt=".", label="Ideal Compton")

    # axs[1].set_ylabel("counts (bin-normalized)")
    axs[1].set_xlim(bins[idx_0mm], bins[-1])
    axs[1].set_ylim(0.0, 1.1)
    axs[1].errorbar(bins[idx_0mm + 2:] - width / 2,
                    hist_0mm_idealcompton[idx_0mm + 1:] / hist_0mm_idealcompton[idx_0mm + 1:],
                    np.sqrt(hist_0mm_idealcompton[idx_0mm + 1:]) / hist_0mm_idealcompton[idx_0mm + 1:],
                    color="black", fmt=".", label="Ideal Compton")
    axs[1].bar(bins[idx_0mm + 2:] - width / 2, hist_0mm_non_nn[idx_0mm + 1:] / hist_0mm_idealcompton[idx_0mm + 1:],
               width=width, align="center", color="tab:green", label="None")
    axs[1].bar(bins[idx_0mm + 2:] - width / 2, hist_0mm_nn[idx_0mm + 1:] / hist_0mm_idealcompton[idx_0mm + 1:],
               bottom=hist_0mm_non_nn[idx_0mm + 1:] / hist_0mm_idealcompton[idx_0mm + 1:],
               width=width, align="center", color="tab:red", label="NN")
    axs[2].errorbar(bins[idx_0mm + 2:] - width / 2,
                    hist_0mm_idealcompton[idx_0mm + 1:] / hist_0mm_idealcompton[idx_0mm + 1:],
                    np.sqrt(hist_0mm_idealcompton[idx_0mm + 1:]) / hist_0mm_idealcompton[idx_0mm + 1:],
                    color="black", fmt=".", label="Ideal Compton")
    axs[2].bar(bins[idx_0mm + 2:] - width / 2, hist_0mm_non_cb[idx_0mm + 1:] / hist_0mm_idealcompton[idx_0mm + 1:],
               width=width, align="center", color="tab:green", label="None")
    axs[2].bar(bins[idx_0mm + 2:] - width / 2, hist_0mm_cb[idx_0mm + 1:] / hist_0mm_idealcompton[idx_0mm + 1:],
               bottom=hist_0mm_non_cb[idx_0mm + 1:] / hist_0mm_idealcompton[idx_0mm + 1:],
               width=width, align="center", color="tab:blue", label="CB")
    axs[2].set_ylabel("")
    axs[2].set_xlim(bins[idx_0mm], bins[-1])
    axs[2].set_ylim(0.0, 1.1)
    axs[3].plot(bins[idx_0mm + 2:] - width / 2,
                hist_0mm_cb[idx_0mm + 1:] / hist_0mm_idealcompton[idx_0mm + 1:] - hist_0mm_nn[
                                                                                  idx_0mm + 1:] / hist_0mm_idealcompton[
                                                                                                  idx_0mm + 1:],
                color="red")
    axs[3].hlines(y=0, xmin=bins[idx_0mm + 1] - width / 2, xmax=bins[-1] - width / 2, linestyles="--", color="black")
    axs[3].set_xlabel("Energy [MeV]")
    axs[3].set_ylabel(r"$\frac{(n_{CB}-n_{NN})}{n_{IC}}$")
    axs[3].set_xlim(bins[idx_0mm], bins[-1])
    fig.tight_layout()

    lblack, _ = axs[0].get_legend_handles_labels()
    lblack.append(mpatches.Patch(color='tab:red', label='NN'))
    lblack.append(mpatches.Patch(color='tab:blue', label='CB'))
    lblack.append(mpatches.Patch(color='tab:green', label='None'))
    axs[0].legend(loc="upper right", handles=lblack)
    if save_plots:
        plt.savefig(dir_plots + "energycut_" + str(energycut) + "_0mm.png")
    else:
        plt.show()


########################################################################################################################

def position_cut(correct=False, save_plots=False, poscut=False):
    import matplotlib.patches as mpatches

    ary_idx = 4
    bins = np.arange(-80, 20, 1.0)
    width = bins[1] - bins[0]  # bins should always have same width

    if correct:
        cb_con = [1, 3]
        nn_con = [2]
    else:
        cb_con = [-1, -3, 1, 3]
        nn_con = [0, 1, 2]

    # energycut
    a = 0.040
    if poscut:
        for i in range(mc_0mm.shape[0]):
            if mc_0mm[i, 8] < -12:
                continue

            rand = np.random.uniform()
            # energy filter
            if rand > 1 / np.exp((mc_0mm[i, 8] + 12) * a):
                mc_0mm[i, 2] = 0
            else:
                continue

    # all selected events are based on their
    ary_0mm_idealcompton = [np.sum(mc_0mm[i, 8]) for i in range(mc_0mm.shape[0]) if (mc_0mm[i, 24])]
    ary_5mm_idealcompton = [np.sum(mc_5mm[i, 8]) for i in range(mc_5mm.shape[0]) if (mc_5mm[i, 24])]

    # CB-selected
    ary_0mm_cb = [mc_0mm[i, 8] for i in range(mc_0mm.shape[0]) if
                  (mc_0mm[i, 24] and mc_0mm[i, 2] in cb_con)]
    ary_5mm_cb = [mc_5mm[i, 8] for i in range(mc_5mm.shape[0]) if
                  (mc_5mm[i, 24] and mc_5mm[i, 2] in cb_con)]
    # NN-selected
    ary_0mm_nn = [mc_0mm[i, 8] for i in range(mc_0mm.shape[0]) if
                  (mc_0mm[i, 24] and mc_0mm[i, 3] in nn_con)]
    ary_5mm_nn = [mc_5mm[i, 8] for i in range(mc_5mm.shape[0]) if
                  (mc_5mm[i, 24] and mc_5mm[i, 3] in nn_con)]
    # Non-selected
    ary_0mm_non_nn = [mc_0mm[i, 8] for i in range(mc_0mm.shape[0]) if
                      (mc_0mm[i, 24] and mc_0mm[i, 3] not in nn_con)]
    ary_5mm_non_nn = [mc_5mm[i, 8] for i in range(mc_5mm.shape[0]) if
                      (mc_5mm[i, 24] and mc_5mm[i, 3] not in nn_con)]

    ary_0mm_non_cb = [mc_0mm[i, 8] for i in range(mc_0mm.shape[0]) if
                      (mc_0mm[i, 24] and mc_0mm[i, 2] not in cb_con)]
    ary_5mm_non_cb = [mc_5mm[i, 8] for i in range(mc_5mm.shape[0]) if
                      (mc_5mm[i, 24] and mc_5mm[i, 2] not in cb_con)]

    # create histograms
    hist_0mm_idealcompton, _ = np.histogram(ary_0mm_idealcompton, bins=bins)
    hist_5mm_idealcompton, _ = np.histogram(ary_5mm_idealcompton, bins=bins)
    hist_0mm_cb, _ = np.histogram(ary_0mm_cb, bins=bins)
    hist_5mm_cb, _ = np.histogram(ary_5mm_cb, bins=bins)
    hist_0mm_nn, _ = np.histogram(ary_0mm_nn, bins=bins)
    hist_5mm_nn, _ = np.histogram(ary_5mm_nn, bins=bins)
    hist_0mm_non_nn, _ = np.histogram(ary_0mm_non_nn, bins=bins)
    hist_5mm_non_nn, _ = np.histogram(ary_5mm_non_nn, bins=bins)
    hist_0mm_non_cb, _ = np.histogram(ary_0mm_non_cb, bins=bins)
    hist_5mm_non_cb, _ = np.histogram(ary_5mm_non_cb, bins=bins)

    # Define x-limits of stacked histograms
    # either the second last bin entry or the first bin with zero entries (due to normalization)
    # yes you can use np.where but that function is dumb
    idx_0mm = 0
    idx_5mm = 0
    for i in range(len(hist_0mm_idealcompton)):
        if hist_0mm_idealcompton[i] == 0:
            idx_0mm = int(i)
            break
    for j in range(len(hist_5mm_idealcompton)):
        if hist_5mm_idealcompton[j] == 0:
            idx_5mm = int(j)
            break

    idx_0mm -= 1

    #######################################################
    # new plot

    # normalized stacked bar plot
    fig, axs = plt.subplots(4, 1, gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    axs[0].set_title("MCSourcePosition.z stacked histogram (0mm)")
    axs[0].set_ylabel("counts")
    axs[0].set_xlim(bins[1], bins[idx_0mm])
    axs[0].errorbar(bins[:idx_0mm] - width / 2, hist_0mm_idealcompton[:idx_0mm],
                    np.sqrt(hist_0mm_idealcompton[:idx_0mm]),
                    color="black", fmt=".", label="Ideal Compton")

    axs[1].set_xlim(bins[1], bins[idx_0mm])
    axs[1].set_ylim(0.0, 1.1)
    axs[1].errorbar(bins[:idx_0mm] - width / 2,
                    hist_0mm_idealcompton[:idx_0mm] / hist_0mm_idealcompton[:idx_0mm],
                    np.sqrt(hist_0mm_idealcompton[:idx_0mm]) / hist_0mm_idealcompton[:idx_0mm],
                    color="black", fmt=".", label="Ideal Compton")
    axs[1].bar(bins[:idx_0mm] - width / 2, hist_0mm_non_nn[:idx_0mm] / hist_0mm_idealcompton[:idx_0mm],
               width=width, align="center", color="tab:green", label="None")
    axs[1].bar(bins[:idx_0mm] - width / 2, hist_0mm_nn[:idx_0mm] / hist_0mm_idealcompton[:idx_0mm],
               bottom=hist_0mm_non_nn[:idx_0mm] / hist_0mm_idealcompton[:idx_0mm],
               width=width, align="center", color="tab:red", label="NN")

    axs[2].set_xlim(bins[1], bins[idx_0mm])
    axs[2].set_ylim(0.0, 1.1)
    axs[2].errorbar(bins[:idx_0mm] - width / 2,
                    hist_0mm_idealcompton[:idx_0mm] / hist_0mm_idealcompton[:idx_0mm],
                    np.sqrt(hist_0mm_idealcompton[:idx_0mm]) / hist_0mm_idealcompton[:idx_0mm],
                    color="black", fmt=".", label="Ideal Compton")
    axs[2].bar(bins[:idx_0mm] - width / 2, hist_0mm_non_cb[:idx_0mm] / hist_0mm_idealcompton[:idx_0mm],
               width=width, align="center", color="tab:green", label="None")
    axs[2].bar(bins[:idx_0mm] - width / 2, hist_0mm_cb[:idx_0mm] / hist_0mm_idealcompton[:idx_0mm],
               bottom=hist_0mm_non_cb[:idx_0mm] / hist_0mm_idealcompton[:idx_0mm],
               width=width, align="center", color="tab:blue", label="CB")

    axs[3].set_xlim(bins[1], bins[idx_0mm])
    axs[3].plot(bins[:idx_0mm] - width / 2,
                hist_0mm_cb[:idx_0mm] / hist_0mm_idealcompton[:idx_0mm] - hist_0mm_nn[
                                                                              :idx_0mm] / hist_0mm_idealcompton[
                                                                                          :idx_0mm],
                color="red")
    axs[3].hlines(y=0, xmin=bins[0] - width / 2, xmax=bins[idx_0mm] - width / 2, linestyles="--", color="black")
    axs[3].set_xlabel("Energy [MeV]")
    axs[3].set_ylabel(r"$\frac{(n_{CB}-n_{NN})}{n_{IC}}$")
    fig.tight_layout()

    lblack, _ = axs[0].get_legend_handles_labels()
    lblack.append(mpatches.Patch(color='tab:red', label='NN'))
    lblack.append(mpatches.Patch(color='tab:blue', label='CB'))
    lblack.append(mpatches.Patch(color='tab:green', label='None'))
    axs[0].legend(loc="upper left", handles=lblack)
    if save_plots:
        plt.savefig(dir_plots + "positioncut_" + str(poscut) + "_0mm.png")
    else:
        plt.show()


########################################################################################################################


# plot_dist_event_selection(8, np.arange(-80, 20, 1.0), "MCPosition_source.z", "position [mm]", save_plots=False)

# stacked histograms previous plots
# plot_dist_stacked(8, np.arange(-80, 20, 1.0), "MCPosition_source.z", "position [mm]", correct=False, save_plots=False)
# plot_dist_stacked(4, np.arange(0.0, 6.0, 0.1), "MC_energy_e", "Energy [MeV]", correct=False, save_plots=True)
# plot_dist_stacked(5, np.arange(0.0, 6.0, 0.1), "MC_energy_p", "Energy [MeV]", correct=False, save_plots=True)
# plot_dist_stacked(20, np.arange(-98.8 / 2 - 1.3 / 2, 98.8 / 2 + 1.3 / 2, 1.3), "MCPosition_e.z", "position [mm]",
#                  correct=False, save_plots=True)

# plot_scatter_energy(4, 5, "MCEnergy_e", "MCEnergy_p", "e Energy [MeV]", "p Energy [MeV]", correct=False, save_plots=False)
# plot_scatter_energy(6, 7, "MCEnergy_e", "MCEnergy_p", "e Energy [MeV]", "p Energy [MeV]", correct=False, save_plots=False)
# plot_scatter_energy_source_pos(correct=False, save_plots=True)

# energy_cut(save_plots=False)
# energy_cut(energycut=True, save_plots=False)

position_cut(save_plots=False)
position_cut(save_plots=False, poscut=True)
