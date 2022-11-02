# header
import numpy as np
import os
import matplotlib.pyplot as plt

# load MCTRUTH files (.npz format)
# define file paths
dir_main = os.getcwd()
dir_data = dir_main + "/data/"

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

#######################################################################

# plot MCTruth source position for CB/NN selected events
# generate histograms
bins_src = np.arange(-80, 20, 1.0)

hist_src_0mm_nn_correct, _ = np.histogram([mc_0mm[i, 8] for i in range(mc_0mm.shape[0]) if mc_0mm[i, 3] in [1, 2]],
                                          bins=bins_src)

hist_src_5mm_nn_correct, _ = np.histogram([mc_5mm[i, 8] for i in range(mc_5mm.shape[0]) if mc_5mm[i, 3] in [1, 2]],
                                          bins=bins_src)

###########################################################################################
# plot distribution MC source position total/ideal compton

# generate histograms
bins_src = np.arange(-80, 20, 1.0)
src_0mm_total = [mc_0mm[i, 8] for i in range(mc_0mm.shape[0]) if mc_0mm[i, 4] != 0.0]
hist_src_0mm_total, _ = np.histogram(src_0mm_total, bins=bins_src)
src_5mm_total = [mc_5mm[i, 8] for i in range(mc_5mm.shape[0]) if mc_5mm[i, 4] != 0.0]
hist_src_5mm_total, _ = np.histogram(src_5mm_total, bins=bins_src)

src_0mm_idealcompton = [mc_0mm[i, 8] for i in range(mc_0mm.shape[0]) if (mc_0mm[i, 24])]
hist_src_0mm_idealcompton, _ = np.histogram(src_0mm_idealcompton, bins=bins_src)
src_5mm_idealcompton = [mc_5mm[i, 8] for i in range(mc_5mm.shape[0]) if (mc_5mm[i, 24])]
hist_src_5mm_idealcompton, _ = np.histogram(src_5mm_idealcompton, bins=bins_src)

plt.figure()
plt.title("MC Source position Ideal Compton Event")
plt.xlabel("z-position [mm]")
plt.ylabel("counts")
# total event histogram
plt.hist(src_0mm_total, bins=bins_src, histtype=u"step", color="black", label="0mm total", density=True, alpha=0.5,
         linestyle="--")
plt.hist(src_5mm_total, bins=bins_src, histtype=u"step", color="red", label="5mm total", density=True, alpha=0.5,
         linestyle="--")
# ideal compton event histogram
plt.hist(src_0mm_idealcompton, bins=bins_src, histtype=u"step", color="black", label="0mm Ideal Compton", density=True)
plt.hist(src_5mm_idealcompton, bins=bins_src, histtype=u"step", color="red", label="5mm Ideal Compton", density=True)
plt.errorbar(bins_src[1:] - 0.5, hist_src_0mm_idealcompton / np.sum(hist_src_0mm_idealcompton),
             np.sqrt(hist_src_0mm_idealcompton) / np.sum(hist_src_0mm_idealcompton), color="black", fmt=".")
plt.errorbar(bins_src[1:] - 0.5, hist_src_5mm_idealcompton / np.sum(hist_src_5mm_idealcompton),
             np.sqrt(hist_src_5mm_idealcompton) / np.sum(hist_src_5mm_idealcompton), color="red", fmt=".")
plt.legend()
plt.grid()
plt.show()

###########################################################################################
# plot distribution MC source position ideal compton / Cut-based

# bins and ideal compton histograms are taken from above
# generated cut-based histograms
src_0mm_cb = [mc_0mm[i, 8] for i in range(mc_0mm.shape[0]) if (mc_0mm[i, 2] != 0.0 and mc_0mm[i, 4] != 0)]
hist_src_0mm_cb, _ = np.histogram(src_0mm_cb, bins=bins_src)
src_5mm_cb = [mc_5mm[i, 8] for i in range(mc_5mm.shape[0]) if (mc_5mm[i, 2] != 0.0 and mc_5mm[i, 4] != 0)]
hist_src_5mm_cb, _ = np.histogram(src_5mm_cb, bins=bins_src)

plt.figure()
plt.title("MC Source position Cut-based")
plt.xlabel("z-position [mm]")
plt.ylabel("counts")
# total event histogram
plt.hist(src_0mm_idealcompton, bins=bins_src, histtype=u"step", color="black", label="0mm ideal compton", density=True,
         alpha=0.5, linestyle="--")
plt.hist(src_5mm_idealcompton, bins=bins_src, histtype=u"step", color="red", label="5mm ideal compton", density=True,
         alpha=0.5, linestyle="--")
# ideal compton event histogram
plt.hist(src_0mm_cb, bins=bins_src, histtype=u"step", color="black", label="0mm Cut-Based", density=True)
plt.hist(src_5mm_cb, bins=bins_src, histtype=u"step", color="red", label="5mm Cut-Based", density=True)
plt.errorbar(bins_src[1:] - 0.5, hist_src_0mm_cb / np.sum(hist_src_0mm_cb),
             np.sqrt(hist_src_0mm_cb) / np.sum(hist_src_0mm_cb), color="black", fmt=".")
plt.errorbar(bins_src[1:] - 0.5, hist_src_5mm_cb / np.sum(hist_src_5mm_cb),
             np.sqrt(hist_src_5mm_cb) / np.sum(hist_src_5mm_cb), color="red", fmt=".")
plt.legend()
plt.grid()
plt.show()

###########################################################################################
# plot distribution MC source position ideal compton / Cut-based

# bins and ideal compton histograms are taken from above
# generated cut-based histograms
src_0mm_nn = [mc_0mm[i, 8] for i in range(mc_0mm.shape[0]) if (mc_0mm[i, 3] in [2])]
hist_src_0mm_nn, _ = np.histogram(src_0mm_nn, bins=bins_src)
src_5mm_nn = [mc_5mm[i, 8] for i in range(mc_5mm.shape[0]) if (mc_5mm[i, 3] in [2])]
hist_src_5mm_nn, _ = np.histogram(src_5mm_nn, bins=bins_src)

plt.figure()
plt.title("MC Source position Neural Network")
plt.xlabel("z-position [mm]")
plt.ylabel("counts")
# total event histogram
plt.hist(src_0mm_idealcompton, bins=bins_src, histtype=u"step", color="black", label="0mm ideal compton", density=True,
         alpha=0.5, linestyle="--")
plt.hist(src_5mm_idealcompton, bins=bins_src, histtype=u"step", color="red", label="5mm ideal compton", density=True,
         alpha=0.5, linestyle="--")
# ideal compton event histogram
plt.hist(src_0mm_nn, bins=bins_src, histtype=u"step", color="black", label="0mm NN", density=True)
plt.hist(src_5mm_nn, bins=bins_src, histtype=u"step", color="red", label="5mm NN", density=True)
plt.errorbar(bins_src[1:] - 0.5, hist_src_0mm_nn / np.sum(hist_src_0mm_nn),
             np.sqrt(hist_src_0mm_nn) / np.sum(hist_src_0mm_nn) / 100, color="black", fmt=".")
plt.errorbar(bins_src[1:] - 0.5, hist_src_5mm_nn / np.sum(hist_src_5mm_nn),
             np.sqrt(hist_src_5mm_nn) / np.sum(hist_src_5mm_nn) / 100, color="red", fmt=".")
plt.legend()
plt.grid()
plt.show()
