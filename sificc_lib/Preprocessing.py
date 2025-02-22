# Some header

import sys
import numpy as np
import uproot
from tqdm import tqdm
from sificc_lib import Event
from sificc_lib.SiFiCC_Module import SiFiCC_Module


class Preprocessing:
    """Processing of root files, generation of neural network input or csv output"""

    def __init__(self, filename):
        # open root file with uproot
        root_file = uproot.open(filename)
        self.__setup(root_file)

        # general tree information
        self.tree = root_file[b'Events']
        self.num_entries = self.tree.numentries

        # cluster information
        self.clusters_count = self.tree['RecoClusterEnergies']
        self.clusters_position = self.tree['RecoClusterPositions.position']
        self.clusters_position_unc = self.tree['RecoClusterPositions.uncertainty']
        self.clusters_energy = self.tree['RecoClusterEnergies.value']
        self.clusters_energy_unc = self.tree['RecoClusterEnergies.uncertainty']
        self.clusters_entries = self.tree['RecoClusterEntries']

    def __setup(self, root_file):
        """
        grab scatterer and absorber setup/dimension from root file
        """
        setup = root_file[b'Setup']

        self.scatterer = SiFiCC_Module(setup['ScattererThickness_x'].array()[0],
                                       setup['ScattererThickness_y'].array()[0],
                                       setup['ScattererThickness_z'].array()[0],
                                       setup["ScattererPosition"].array()[0])
        self.absorber = SiFiCC_Module(setup['AbsorberThickness_x'].array()[0],
                                      setup['AbsorberThickness_y'].array()[0],
                                      setup['AbsorberThickness_z'].array()[0],
                                      setup['AbsorberPosition'].array()[0])

    def iterate_events(self, n=None, basket_size=100000, desc='processing root file', bar_update_size=1000):
        """
        Iteration through all events within a root file.
        Iteration is done stepwise via root baskets to not overload the memory.
        """
        # check if n is smaller than the number of entries in the root file
        # else set entrystop to None to iterate the full root file
        # adjust total entries for bar progression
        bar_total = self.num_entries
        if n is not None:
            if n > self.num_entries:
                n = None
                bar_total = self.num_entries
            elif (n > 0 and n < self.num_entries):
                bar_total = n

        # define progress bar
        prog_bar = tqdm(total=bar_total, ncols=100, file=sys.stdout, desc=desc)
        bar_step = 0
        for start, end, basket in self.tree.iterate(Event.l_leaves, entrysteps=basket_size,
                                                    reportentries=True, namedecode='utf-8',
                                                    entrystart=0, entrystop=n):
            length = end - start
            for idx in range(length):
                yield self.__event_at_basket(basket, idx)

                bar_step += 1
                if bar_step % bar_update_size == 0:
                    prog_bar.update(bar_update_size)

        prog_bar.update(self.num_entries % bar_update_size)
        prog_bar.close()

    def __event_at_basket(self, basket, position):
        """
        grab event from a root basket at a given position
        """

        event = Event(EventNumber=basket["EventNumber"][position],
                      MCEnergy_Primary=basket['MCEnergy_Primary'][position],
                      MCEnergy_e=basket['MCEnergy_e'][position],
                      MCEnergy_p=basket['MCEnergy_p'][position],
                      MCPosition_e=basket['MCPosition_e'][position],
                      MCInteractions_e=basket['MCInteractions_e'][position],
                      MCPosition_p=basket['MCPosition_p'][position],
                      MCInteractions_p=basket['MCInteractions_p'][position],
                      MCPosition_source=basket['MCPosition_source'][position],
                      MCDirection_source=basket['MCDirection_source'][position],
                      MCComptonPosition=basket['MCComptonPosition'][position],
                      MCDirection_scatter=basket['MCDirection_scatter'][position],
                      Identified=basket['Identified'][position],
                      RecoClusterEnergies=basket['RecoClusterEnergies'][position],
                      RecoClusterPosition=basket['RecoClusterPositions.position'][position],
                      RecoClusterPosition_uncertainty=basket['RecoClusterPositions.uncertainty'][position],
                      RecoClusterEnergies_values=basket['RecoClusterEnergies.value'][position],
                      RecoClusterEnergies_uncertainty=basket['RecoClusterEnergies.uncertainty'][position],
                      RecoClusterEntries=basket['RecoClusterEntries'][position],
                      MCSimulatedEventType=basket['MCSimulatedEventType'][position],
                      scatterer=self.scatterer,
                      absorber=self.absorber)
        return event

    def get_event(self, position):
        """Return event for a given position in the root file"""
        for basket in self.tree.iterate(Event.l_leaves, entrystart=position, entrystop=position + 1,
                                        namedecode='utf-8'):
            return self.__event_at_basket(basket, 0)

