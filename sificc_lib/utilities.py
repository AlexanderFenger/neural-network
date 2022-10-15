import numpy as np


class utilities:

    def show_root_file_analysis(simulation, only_valid=True):
        import matplotlib.pyplot as plt
        n_distributed_clusters = 0
        n_compton = 0
        n_complete_compton = 0
        n_complete_distributed_compton = 0
        n_ideal_compton = 0
        n_ep = 0
        n_pe = 0
        n_matching_ideal_compton = 0
        n_overlap_matching_ideal = 0
        l_matching_idx = []

        for event in simulation.iterate_events():
            if not event.is_valid and only_valid:
                continue
            n_distributed_clusters += 1 if event.is_valid else 0
            n_compton += 1 if event.is_compton else 0
            n_complete_compton += 1 if event.is_compton_full else 0
            n_complete_distributed_compton += 1 if event.is_complete_distributed_compton else 0
            n_ideal_compton += 1 if event.is_ideal_compton else 0
            n_ep += 1 if event.is_ep else 0
            n_pe += 1 if event.is_pe else 0

            if event.is_ideal_compton:
                if event.is_clusters_matching:
                    n_matching_ideal_compton += 1
                    event._sort_clusters()
                    l_matching_idx.append(event._arg_matching_cluster(event.real_p_position))
                    l_matching_idx.append(event._arg_matching_cluster(event.real_e_position))
                n_overlap_matching_ideal += 1 if event.is_clusters_overlap else 0

        print('{:8,d} total entries'.format(simulation.num_entries))
        print('{:8,d} valid entries with distrbuted clusters'.format(n_distributed_clusters))
        print('{:8,d} compton events'.format(n_compton))
        print('{:8,d} compton + second interaction'.format(n_complete_compton))
        print('{:8,d} compton + second in different module'.format(n_complete_distributed_compton))
        print('{:8,d} ideal compton events'.format(n_ideal_compton))
        print('\t{:8,d} ep'.format(n_ep))
        print('\t{:8,d} pe'.format(n_pe))
        print('{:8,d} ideal compton events with matching clusters'.format(n_matching_ideal_compton))
        print('{:8,d} ideal compton events with overlapping clusters'.format(n_overlap_matching_ideal))
        n, bins, _ = plt.hist(l_matching_idx, np.arange(0, np.max(l_matching_idx) + 2))
        plt.xticks(np.arange(0, np.max(l_matching_idx) + 1), np.arange(1, np.max(l_matching_idx) + 2))
        plt.xlabel('argmax of electron and photon clusters')
        plt.ylabel('count')
        plt.show()
        print('histogram bars\' count:', n)

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
            print("{} | {} | ({:7.3f}, {:7.3f}, {:7.3f}) | {:3} | Module: {}".format(i,
                                                                                     0,
                                                                                     cluster.x,
                                                                                     cluster.y,
                                                                                     cluster.z,
                                                                                     event.RecoClusterEntries[i],
                                                                                     event.cluster_module(cluster)))
