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
    l_leaves = ['MCEnergy_Primary',
                'MCEnergy_e',
                'MCEnergy_p',
                'MCPosition_source',
                'MCSimulatedEventType',
                'MCDirection_source',
                'MCComptonPosition',
                'MCDirection_scatter',
                'MCPosition_e',
                'MCInteractions_e',
                'MCPosition_p',
                'MCInteractions_p',
                'Identified',
                'PurCrossed',
                'RecoClusterPositions.position',
                'RecoClusterPositions.uncertainty',
                'RecoClusterEnergies',
                'RecoClusterEnergies.value',
                'RecoClusterEnergies.uncertainty',
                'RecoClusterEntries']

    def __init__(self, MCEnergy_Primary, MCEnergy_e, MCEnergy_p, MCPosition_source, MCSimulatedEventType,
                 MCDirection_source, MCComptonPosition, MCDirection_scatter, MCPosition_e, MCInteractions_e,
                 MCPosition_p, MCInteractions_p, Identified, PurCrossed, RecoClusterPosition,
                 RecoClusterPosition_uncertainty, RecoClusterEnergies, RecoClusterEnergies_values,
                 RecoClusterEnergies_uncertainty,
                 RecoClusterEntries, scatterer, absorber):

        # define the main values of a simulated event
        self.MCEnergy_Primary = MCEnergy_Primary
        self.MCEnergy_e = MCEnergy_e
        self.MCEnergy_p = MCEnergy_p
        self.MCPosition_source = MCPosition_source
        self.MCSimulatedEventType = MCSimulatedEventType
        self.MCDirection_source = MCDirection_source
        self.MCComptonPosition = MCComptonPosition
        self.MCDirection_scatter = MCDirection_scatter
        self.MCPosition_e = MCPosition_e
        self.MCInteractions_e = MCInteractions_e
        self.MCPosition_p = MCPosition_p
        self.MCInteractions_p = MCInteractions_p
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
        self.is_compton_complete = False  # compton event and 2 cluster in absorber
        self.MCPosition_e_first = TVector3(0, 0, 0)
        self.MCPosition_p_first = TVector3(0, 0, 0)
        self.is_compton_ideal = False  # event is an ideal compton event

        # check if the event is a valid event by considering the clusters associated with it
        # the event is considered valid if there is at least one cluster within each module of the SiFiCC
        if (self.RecoClusterEnergies >= 2
                and scatterer.is_any_cluster_inside(self.RecoClusterPosition)
                and absorber.is_any_cluster_inside(self.RecoClusterPosition)):
            self.is_valid = True
        else:
            self.is_valid = False

        # check if the event is a Compton event
        self.is_compton = True if self.MCEnergy_e >= 0 else False

        # check if event is a complete compton event
        # pre-condition: event is a compton event
        # p goes through a second interaction and event type is restricted:
        # 0 < p interaction < 10 (???)
        # 10 <= e interaction < 20 (???)
        if (self.is_compton
                and len(self.MCPosition_e) >= 2
                and len(self.MCPosition_e) >= 1
                and ((self.MCInteractions_p[1:] > 0) & (self.MCInteractions_p[1:])).any()
                and ((self.MCInteractions_e[0] >= 10) & (self.MCInteractions_e[0] < 20))):
            self.is_compton_complete = True
        else:
            self.is_compton_complete = False

        # initialize e & p first interaction position
        if self.is_compton_complete:
            for idx in range(1, len(self.MCInteractions_p)):
                if 0 < self.MCInteractions_p[idx] < 10:
                    self.MCPosition_p_first = self.MCPosition_p[idx]
                    break
            for idx in range(0, len(self.MCInteractions_e)):
                if 10 <= self.MCInteractions_e[idx] < 20:
                    self.MCPosition_e_first = self.MCPosition_e[idx]
                    break
        else:
            self.MCPosition_p_first = TVector3(0, 0, 0)
            self.MCPosition_e_first = TVector3(0, 0, 0)

        # check if the event is an ideal Compton event and what type is it (EP or PE)
        # ideal Compton event = complete distributed Compton event where the next interaction of both
        # e and p is in the different modules of SiFiCC
        if self.is_compton_complete \
                and scatterer.is_point_inside_x(self.MCPosition_e_first) \
                and absorber.is_point_inside_x(self.MCPosition_p_first) \
                and self.MCSimulatedEventType == 2:
            self.is_compton_ideal = True
            self.is_ep = True
            self.is_pe = False
        elif self.is_compton_complete \
                and scatterer.is_point_inside_x(self.MCPosition_p_first) \
                and absorber.is_point_inside_x(self.MCPosition_e_first) \
                and self.MCSimulatedEventType == 2:
            self.is_compton_ideal = True
            self.is_ep = False
            self.is_pe = True
        else:
            self.is_compton_ideal = False
            self.is_ep = False
            self.is_pe = False

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
