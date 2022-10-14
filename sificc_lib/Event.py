# Author: Awal Awal
# Date: Jul 2020
# Email: awal.nova@gmail.com

import math
import numpy as np
from sificc_lib import utilities
from uproot_methods.classes.TVector3 import TVector3


class Event:
    """Represents a single event in a ROOT file"""

    # list of leaves that are required from a ROOT file to properly instantiate an Event object
    l_leaves = ['Energy_Primary',
                'RealEnergy_e',  # Gen: Energy electron
                'RealEnergy_p',  # Gen:Energy proton
                'RealPosition_source',  # Gen: position source
                # 'SimulatedEventType',  # Gen: Event Tag
                'RealDirection_source',
                'RealComptonPosition',  # Gen: position of compton event (in scatterer?)
                'RealDirection_scatter',
                'RealPosition_e',
                'RealInteractions_e',
                'RealPosition_p',
                'RealInteractions_p',
                'Identified',
                'PurCrossed',
                'RecoClusterPositions.position',
                'RecoClusterPositions.uncertainty',
                'RecoClusterEnergies',
                'RecoClusterEnergies.value',
                'RecoClusterEnergies.uncertainty',
                'RecoClusterEntries']

    def __init__(self, real_primary_energy, real_e_energy, real_p_energy, real_e_positions,
                 real_e_interactions, real_p_positions, real_p_interactions, real_src_pos, real_src_dir,
                 real_compton_pos, real_scatter_dir, identification_code, crossed, clusters_count,
                 clusters_position, clusters_position_unc, clusters_energy, clusters_energy_unc,
                 clusters_entries, event_type,
                 scatterer, absorber):

        # define the main values of a simulated event
        self.event_type = event_type
        self.real_primary_energy = real_primary_energy
        self.real_e_energy = real_e_energy
        self.real_p_energy = real_p_energy
        self.real_e_position = real_e_positions
        self.real_e_interaction = real_e_interactions
        self.real_p_position = real_p_positions
        self.real_p_interaction = real_p_interactions
        self.real_src_pos = real_src_pos
        self.real_src_dir = real_src_dir
        self.real_compton_pos = real_compton_pos
        self.real_scatter_dir = real_scatter_dir
        self.identification_code = identification_code
        self.crossed = crossed
        self.clusters_count = clusters_count  # given by 'RecoClusterEnergies'
        self.clusters_position = clusters_position
        self.clusters_position_unc = clusters_position_unc
        self.clusters_energy = clusters_energy
        self.clusters_energy_unc = clusters_energy_unc
        self.clusters_entries = clusters_entries

        # tags for further analysis
        self.is_valid = False  # at least 2 clusters, one in each module
        self.is_compton = False  # compton scattering occurred (checked by positive electron energy)
        self.is_compton_full = False  # compton event and 2 cluster in absorber

        # check if the event is a valid event by considering the clusters associated with it
        # the event is considered valid if there is at least one cluster within each module of the SiFiCC
        if self.clusters_count >= 2 \
                and scatterer.is_any_point_inside_x(self.clusters_position) \
                and absorber.is_any_point_inside_x(self.clusters_position):
            self.is_valid = True
        else:
            self.is_valid = False

        # check if the event is a Compton event
        self.is_compton = True if self.real_e_energy != 0 else False

        # check if the event is a complete Compton event
        # complete Compton event= Compton event + 1 e and 2 p interactions in which
        # 0 < p interaction < 10
        # 10 <= e interaction < 20
        # Note: first interaction of p is the compton event
        if self.is_compton \
                and len(self.real_p_position) >= 2 \
                and len(self.real_e_position) >= 1 \
                and ((self.real_p_interaction[1:] > 0) & (self.real_p_interaction[1:] < 10)).any() \
                and ((self.real_e_interaction[0] >= 10) & (self.real_e_interaction[0] < 20)):
            self.is_compton_full = True
        else:
            self.is_compton_full = False

        # initialize e & p first interaction position
        if self.is_compton_full:
            for idx in range(1, len(self.real_p_interaction)):
                if 0 < self.real_p_interaction[idx] < 10:
                    self.real_p_position = self.real_p_position[idx]
                    break
            for idx in range(0, len(self.real_e_interaction)):
                if 10 <= self.real_e_interaction[idx] < 20:
                    self.real_e_position = self.real_e_position[idx]
                    break
        """
        else:
            self.real_p_position = TVector3(0, 0, 0)
            self.real_e_position = TVector3(0, 0, 0)
        """

