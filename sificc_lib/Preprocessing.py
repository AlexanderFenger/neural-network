# Some header

import sys
import uproot
import numpy as np
from tqdm import tqdm
from sificc_lib import root_files, Event, SiFiCC_Module


class Preprocessing:
    """Processing of root files, generation of neural network input or csv output"""

    # TODO: Processing of root files
    # TODO: export to csv
    # TODO: Preprocessing of data for neural network input

    def __init__(self, filename):
        # open root file with uproot
        root_file = uproot.open(filename)

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
                                       setup['ScattererPosition'].array()[0])
        self.absorber = SiFiCC_Module(setup['AbsorberThickness_x'].array()[0],
                                      setup['AbsorberThickness_y'].array()[0],
                                      setup['AbsorberThickness_z'].array()[0],
                                      setup['AbsorberPosition'].array()[0])

    def iterate_events(self, basket_size=100000, desc='processing root file', bar_update_size=1000):
        """
        Iteration through all events within a root file.
        Iteration is done stepwise via root baskets to not overload the memory.
        """
        total = self.num_entries
        # define progress bar
        prog_bar = tqdm(total=total, ncols=100, file=sys.stdout, desc=desc)
        bar_step = 0
        for start, end, basket in self.tree.iterate(Event.l_leaves, entrysteps=basket_size,
                                                    reportentries=True, namedecode='utf-8',
                                                    entrystart=0, entrystop=None):
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

        event = Event(real_primary_energy=basket['Energy_Primary'][position],
                      real_e_energy=basket['RealEnergy_e'][position],
                      real_p_energy=basket['RealEnergy_p'][position],
                      real_e_positions=basket['RealPosition_e'][position],
                      real_e_interactions=basket['RealInteractions_e'][position],
                      real_p_positions=basket['RealPosition_p'][position],
                      real_p_interactions=basket['RealInteractions_p'][position],
                      real_src_pos=basket['RealPosition_source'][position],
                      real_src_dir=basket['RealDirection_source'][position],
                      real_compton_pos=basket['RealComptonPosition'][position],
                      real_scatter_dir=basket['RealDirection_scatter'][position],
                      identification_code=basket['Identified'][position],
                      crossed=basket['PurCrossed'][position],
                      clusters_count=basket['RecoClusterEnergies'][position],
                      clusters_position=basket['RecoClusterPositions.position'][position],
                      clusters_position_unc=basket['RecoClusterPositions.uncertainty'][position],
                      clusters_energy=basket['RecoClusterEnergies.value'][position],
                      clusters_energy_unc=basket['RecoClusterEnergies.uncertainty'][position],
                      clusters_entries=basket['RecoClusterEntries'][position],
                      event_type=basket['SimulatedEventType'][position],
                      scatterer=self.scatterer,
                      absorber=self.absorber)
        return event

    def export_to_csv(self, only_valid=True, n=0):

        # iterate through all events
        for event in self.iterate_events():
            if not event.is_distributed_clusters and only_valid:
                continue

            # grab all informations from