import numpy as np
import matplotlib.pyplot as plt
import Preprocessing


class Plotter:
    def __init__(self, dir_plot):
        self.dir_plot = dir_plot  # the target directory for plot PDFs

    def plot_source_dist(self, preprocessing: Preprocessing, filename):
        """plot the source distribution as a 2D scatter plot"""

        list_x = []
        list_y = []

        for event in preprocessing.iterate_events(n=10000):
            list_x.append(event.RealPosition_source.x)
            list_y.append(event.RealPosition_source.y)

        # Scatter plot
        plt.figure()
        plt.hist2d(list_x, list_y)
        plt.savefig(self.dir_plot + filename + ".pdf")
