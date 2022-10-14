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

    def print_event_summary(preprocessing):
        """Prints out a summary of one random event"""
        # choose random event
        n = 42

        event = preprocessing.get_event(position=n)

        print("\nPrinting event summary\n")
        print("Event number = ", n)
        print("Event type = ", event.event_type, type(event.event_type))
        print("Energy_primary = ", event.real_primary_energy, type(event.real_primary_energy))
