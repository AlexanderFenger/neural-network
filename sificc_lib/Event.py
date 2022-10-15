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
                'RealEnergy_e',
                'RealEnergy_p',
                'RealPosition_source',
                'SimulatedEventType',
                'RealDirection_source',
                'RealComptonPosition',
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

    def __init__(self, Energy_Primary, RealEnergy_e, RealEnergy_p, RealPosition_source, SimulatedEventType,
                 RealDirection_source, RealComptonPosition, RealDirection_scatter, RealPosition_e, RealInteraction_e,
                 RealPosition_p, RealInteraction_p, Identified, PurCrossed, RecoClusterPosition,
                 RecoClusterPosition_uncertainty, RecoClusterEnergies, RecoClusterEnergies_values,
                 RecoClusterEnergies_uncertainty,
                 RecoClusterEntries, scatterer, absorber):

        # define the main values of a simulated event
        self.Energy_Primary = Energy_Primary
        self.RealEnergy_e = RealEnergy_e
        self.RealEnergy_p = RealEnergy_p
        self.RealPosition_source = RealPosition_source
        self.SimulatedEventType = SimulatedEventType
        self.RealDirection_source = RealDirection_source
        self.RealComptonPosition = RealComptonPosition
        self.RealDirection_scatter = RealDirection_scatter
        self.RealPosition_e = RealPosition_e
        self.RealInteraction_e = RealInteraction_e
        self.RealPosition_p = RealPosition_p
        self.RealInteraction_p = RealInteraction_p
        self.Identified = Identified
        self.PurCrossed = PurCrossed
        self.RecoClusterPosition = RecoClusterPosition
        self.RecoClusterPosition_uncertainty = RecoClusterPosition_uncertainty
        self.RecoClusterEnergies = RecoClusterEnergies  # <=> cluster counts
        self.RecoClusterEnergies_values = RecoClusterEnergies_values
        self.RecoClusterEnergies_uncertainty = RecoClusterEnergies_uncertainty
        self.RecoClusterEntries = RecoClusterEntries
        self.scatterer = scatterer
        self.absorber = absorber

        # tags for further analysis
        self.is_valid = False  # at least 2 clusters, one in each module
        self.is_compton = False  # compton scattering occurred (checked by positive electron energy)
        self.is_compton_full = False  # compton event and 2 cluster in absorber

        # check if the event is a valid event by considering the clusters associated with it
        # the event is considered valid if there is at least one cluster within each module of the SiFiCC
        if (self.RecoClusterEnergies >= 2
                and scatterer.is_any_cluster_inside(self.RecoClusterPosition)
                and absorber.is_any_cluster_inside(self.RecoClusterPosition)):
            self.is_valid = True
        else:
            self.is_valid = False

        # check if the event is a Compton event
        self.is_compton = True if self.RealEnergy_e >= 0 else False

        # check if the event is a complete Compton event
        # complete Compton event= Compton event + 1 e and 2 p interactions in which
        # 0 < p interaction < 10
        # 10 <= e interaction < 20
        # Note: first interaction of p is the compton event
        if self.is_compton \
                and len(self.RealPosition_p) >= 2 \
                and len(self.RealPosition_e) >= 1 \
                and ((self.RealInteraction_e[1:] > 0) & (self.RealInteraction_p[1:] < 10)).any() \
                and ((self.RealInteraction_e[0] >= 10) & (self.RealInteraction_e[0] < 20)):
            self.is_compton_full = True
        else:
            self.is_compton_full = False

    ####################################################################################################################

    def cluster_module(self, cluster, return_int=False):
        """
        returns if cluster is in scatterer or absorber
        return_int: 0: None; 1: Scatterer; 2: Absorber
        """
        if self.scatterer.is_cluster_inside(cluster):
            if return_int:
                return 1
            else:
                return "Scatterer"
        if self.absorber.is_cluster_inside(cluster):
            if return_int:
                return 2
            else:
                return "Absorber"
        # else
        if return_int:
            return 0
        else:
            return "None"
