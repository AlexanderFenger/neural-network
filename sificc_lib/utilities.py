import numpy as np


class utilities:

    def print_event_summary(preprocessing, n=0):
        """Prints out a summary of one random event"""

        # grab event from root tree
        event = preprocessing.get_event(position=n)

        print("\nPrinting event summary\n")
        print("Event number: ", n)
        print("Event type: {}".format(event.MCSimulatedEventType))
        print("--------------")
        print("EnergyPrimary: {:.3f}".format(event.MCEnergy_Primary))
        print("RealEnergy_e: {:.3f}".format(event.MCEnergy_e))
        print("RealEnergy_p: {:.3f}".format(event.MCEnergy_p))
        print("RealPosition_source: ({:7.3f}, {:7.3f}, {:7.3f})".format(event.MCPosition_source.x,
                                                                        event.MCPosition_source.y,
                                                                        event.MCPosition_source.z))
        print("RealDirection_source: ({:7.3f}, {:7.3f}, {:7.3f})".format(event.MCDirection_source.x,
                                                                         event.MCDirection_source.y,
                                                                         event.MCDirection_source.z))
        print("RealComptonPosition: ({:7.3f}, {:7.3f}, {:7.3f})".format(event.MCComptonPosition.x,
                                                                        event.MCComptonPosition.y,
                                                                        event.MCComptonPosition.z))
        print("RealDirection_scatter: ({:7.3f}, {:7.3f}, {:7.3f})".format(event.MCDirection_scatter.x,
                                                                          event.MCDirection_scatter.y,
                                                                          event.MCDirection_scatter.z))
        print("\nRealPosition_e / RealInteractions_e:")
        for i in range(len(event.MCInteractions_e)):
            print("({:7.3f}, {:7.3f}, {:7.3f}) | {}".format(event.MCPosition_e.x[i],
                                                            event.MCPosition_e.y[i],
                                                            event.MCPosition_e.z[i],
                                                            event.MCInteractions_e[i]))
        print("\nRealPosition_p / RealInteractions_p:")
        for i in range(len(event.MCInteractions_p)):
            print("({:7.3f}, {:7.3f}, {:7.3f}) | {}".format(event.MCPosition_p.x[i],
                                                            event.MCPosition_p.y[i],
                                                            event.MCPosition_p.z[i],
                                                            event.MCInteractions_p[i]))

        print("\n Cluster Entries: ")
        print("Energy / Position / Entries / Module")
        for i, cluster in enumerate(event.RecoClusterPosition):
            print("{:.3f} | {} | ({:7.3f}, {:7.3f}, {:7.3f}) | {:3} ".format(i,
                                                                             event.RecoClusterEnergies_values[i],
                                                                             cluster.x,
                                                                             cluster.y,
                                                                             cluster.z,
                                                                             event.RecoClusterEntries[i]))
