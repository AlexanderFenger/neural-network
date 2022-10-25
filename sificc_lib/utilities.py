import numpy as np


class utilities:


    def print_event_summary(preprocessing, n=0):
        """Prints out a summary of one random event"""

        # grab event from root tree
        event = preprocessing.get_event(position=n)

        print("\nPrinting event summary\n")
        print("Event number: ", n)
        print("Event type: {}".format(event.SimulatedEventType))
        print("--------------")
        print("EnergyPrimary: {:.3f}".format(event.Energy_Primary))
        print("RealEnergy_e: {:.3f}".format(event.RealEnergy_e))
        print("RealEnergy_p: {:.3f}".format(event.RealEnergy_p))
        print("RealPosition_source: ({:7.3f}, {:7.3f}, {:7.3f})".format(event.RealPosition_source.x,
                                                                        event.RealPosition_source.y,
                                                                        event.RealPosition_source.z))
        print("RealDirection_source: ({:7.3f}, {:7.3f}, {:7.3f})".format(event.RealDirection_source.x,
                                                                         event.RealDirection_source.y,
                                                                         event.RealDirection_source.z))
        print("RealComptonPosition: ({:7.3f}, {:7.3f}, {:7.3f})".format(event.RealComptonPosition.x,
                                                                        event.RealComptonPosition.y,
                                                                        event.RealComptonPosition.z))
        print("RealDirection_scatter: ({:7.3f}, {:7.3f}, {:7.3f})".format(event.RealDirection_scatter.x,
                                                                          event.RealDirection_scatter.y,
                                                                          event.RealDirection_scatter.z))
        print("\nRealPosition_e / RealInteractions_e:")
        for i in range(len(event.RealInteractions_e)):
            print("({:7.3f}, {:7.3f}, {:7.3f}) | {}".format(event.RealPosition_e.x[i],
                                                            event.RealPosition_e.y[i],
                                                            event.RealPosition_e.z[i],
                                                            event.RealInteractions_e[i]))
        print("\nRealPosition_p / RealInteractions_p:")
        for i in range(len(event.RealInteractions_p)):
            print("({:7.3f}, {:7.3f}, {:7.3f}) | {}".format(event.RealPosition_p.x[i],
                                                            event.RealPosition_p.y[i],
                                                            event.RealPosition_p.z[i],
                                                            event.RealInteractions_p[i]))

        print("\n Cluster Entries: ")
        print("Energy / Position / Entries / Module")
        for i, cluster in enumerate(event.RecoClusterPosition):
            print("{:.3f} | {} | ({:7.3f}, {:7.3f}, {:7.3f}) | {:3} | Module: {}".format(i,
                                                                                         event.RecoClusterEnergies_values[i],
                                                                                         cluster.x,
                                                                                         cluster.y,
                                                                                         cluster.z,
                                                                                         event.RecoClusterEntries[i],
                                                                                         event.cluster_module(cluster)))
