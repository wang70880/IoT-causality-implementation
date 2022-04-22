from src.event_processing import Hprocessor
import collections
import itertools
import numpy as np
class Evaluator():

    def __init__(self, dataset, partition_config, tau_max) -> None:
        self.dataset = dataset
        self.partition_config = partition_config
        self.tau_max = tau_max
        self.event_processor = Hprocessor(dataset)
        self.temporal_pair_dict = None
        self.spatial_array = None

        self.user_correlation_dict = None
        self.physical_correlation_dict = None
        self.automation_correlation_dict = None
    
    def _temporal_pair_identification(self):
        """Identification of the temporal correlation pair in the following format.

            (attr_c, attr_o, E[attr_o|attr_c=0], E[attr_o|attr_c=1])

            These pairs are indexed by the time lag tau and the partitioning scheme (i.e., the date).

        Args:
            partition_config (tuple): _description_
            tau_max (int, optional): _description_. Defaults to 1.
        Returns:

        """
        temporal_pair_dict = {} # First index: frame_id. Second index: lag. Value: an integer array of shape num_attrs X num_attrs
        for frame_id in range( len(self.event_processor.frame_dict.keys())): # Collect all lagged pairs in all frames
            temporal_pair_dict[frame_id] = {}
            attr_names = self.event_processor.attr_names; num_attrs = len(attr_names)
            attr_sequence = self.event_processor.frame_dict[frame_id]['attr-sequence']
            state_sequence = self.event_processor.frame_dict[frame_id]['state-sequence']
            assert(len(attr_sequence) == len(state_sequence))
            num_event = len(attr_sequence)
            for event_id in range(num_event): # Count the occurrence of each attr pair and update the corresponding array
                for lag in range (1, self.tau_max + 1):
                    if event_id + lag >= num_event:
                        continue
                    temporal_pair_dict[frame_id][lag] = np.zeros(shape=(num_attrs, num_attrs), dtype=np.int64) if lag not in temporal_pair_dict[frame_id].keys() else temporal_pair_dict[frame_id][lag]
                    prior_attr = attr_sequence[event_id]; con_attr = attr_sequence[event_id + lag]
                    temporal_pair_dict[frame_id][lag][attr_names.index(prior_attr), attr_names.index(con_attr)] += 1

        for frame_id in range( len(self.event_processor.frame_dict.keys())):
            for lag in range (1, self.tau_max + 1):
                attr_array = temporal_pair_dict[frame_id][lag]
                pair_sum = np.sum(attr_array)
                for idx, x in np.ndenumerate(attr_array):
                    attr_array[idx] = 0 if x < pair_sum * 0.001 else 1 # NOTE: Filter out pairs with low frequencies (threshold: %1 of the total frequency)

        self.temporal_pair_dict = temporal_pair_dict

    def _spatial_correlation_identification(self):
        attr_names = self.event_processor.attr_names; num_attrs = len(attr_names)
        spatial_array = np.zeros(shape=(num_attrs, num_attrs), dtype=np.int64)
        if self.dataset == 'hh101':
            # hard-coded pairs...
            area_list = [[['T102', 'D002'], ['M001', 'LS001'], ['M010', 'LS010']],\
                               [['M001', 'LS001'], ['D003', 'T103'], ['M011', 'LS011']],\
                                [['D003', 'T103'], ['MA015', 'LS015']],\
                                [['M011', 'LS011'], ['M009', 'LS009'], ['MA014', 'LS014'], ['M012', 'LS012']],\
                                [['M001', 'LS001'], ['M010', 'LS010']],\
                                [['M010', 'LS010'], ['M005', 'LS005'], ['D001', 'T101'], ['T104', 'T105']], \
                                [['M005', 'LS005'], ['MA013', 'LS013'], ['M008', 'LS008']],\
                                [['M005', 'LS005'], ['M004', 'LS004']],\
                                [['M004', 'LS004'], ['MA016', 'LS016'], ['M003', 'LS003']],\
                                [['M003', 'LS003'], ['M002', 'LS002'], ['M007', 'LS007'], ['M006', 'LS006']]]
        for area in area_list: # For any two device_lists in the area, construct the array
            # First identify the closeness for in each device list (e.g., T101 and D002)
            adherent_attrs = sum(area, [])
            attr_pairs = [(x, y) for x in adherent_attrs for y in adherent_attrs]
            for attr_pair in attr_pairs:
                if attr_pair[0] in attr_names and attr_pair[1] in attr_names:
                    spatial_array[attr_names.index(attr_pair[0]), attr_names.index(attr_pair[1])] = 1
        
        self.spatial_array = spatial_array

    def _physical_correlation_identification(self):
        pass

    def _automation_correlation_identification(self):
        pass

    def construct_golden_standard(self):
        self.event_processor.initiate_data_preprocessing(partition_config=self.partition_config)
        self._temporal_pair_identification()
        self._spatial_correlation_identification()
        self.user_correlation_dict = {}
        for frame_id in range(self.event_processor.frame_count):
            self.user_correlation_dict[frame_id] = {}
            for tau in range(1, self.tau_max + 1):
                temporal_array = self.temporal_pair_dict[frame_id][tau]
                user_correlation_array = temporal_array * self.spatial_array # The user correlation should be satisfy the temporal and spatial coherence.
                self.user_correlation_dict[frame_id][tau] = user_correlation_array
        # test_user_correlation_array = self.user_correlation_dict[0][1]
        # for idx, x in np.ndenumerate(test_user_correlation_array):
        #     if x == 1 and self.event_processor.attr_names[idx[0]] != self.event_processor.attr_names[idx[1]]:
        #         print("{} -> {}".format(self.event_processor.attr_names[idx[0]], self.event_processor.attr_names[idx[1]]))

    def _estimate_single_discovery_accuracy(self, frame_id, tau, pc_results):
        n_discovery = len( [x for x in sum(list(pc_results.values()), []) if x[1] == -1 * tau ] ) # The number of discovered causal links
        truth_array = self.user_correlation_dict[frame_id][tau]; truth_count = np.sum(truth_array) # The number of true causal links
        tp = 0; fn = 0; fp = 0
        for idx, x in np.ndenumerate(truth_array):
            if x == 1: # We only check those causal links
                if (idx[0] in pc_results.keys()) and ( (idx[1], -1 * tau) in pc_results[idx[0]]):
                    tp += 1
                else:
                    fn += 1
                    print("Missing {} -> {}".format( self.event_processor.attr_names[idx[0]], self.event_processor.attr_names[idx[1]] ))

        fp = n_discovery - tp
        print("\n##\n## [frame_id={}, tau={}] Evaluating stable-pc algorithm\n##".format(frame_id, tau)
                  + "\n\nParameters:")
        print(  "\nn_discovery = %d" % n_discovery
                  + "\ntruth_count = %s" % truth_count 
                  + "\ntp = %d" % tp
                  + "\nfn = %d" % fn 
                  + "\nfp = %d" % fp)
if __name__ == '__main__':
    partition_config = (1, 10)
    tau_max = 1
    evaluator = Evaluator(dataset='hh101', partition_config=partition_config, tau_max=tau_max)
    evaluator.construct_golden_standard()
    pc_results = {0: [(0, -1), (1, -1), (4, -1), (3, -1), (17, -1), (15, -1), (27, -1), (30, -1), (10, -1), (13, -1), (12, -1), (18, -1), (14, -1), (2, -1), (6, -1), (11, -1), (33, -1), (34, -1), (5, -1), (25, -1), (32, -1), (9, -1), (29, -1), (26, -1), (20, -1), (22, -1)], 1: [(1, -1), (18, -1), (4, -1), (3, -1), (8, -1), (30, -1), (17, -1), (0, -1), (27, -1), (25, -1), (31, -1), (29, -1), (13, -1), (33, -1), (34, -1), (10, -1), (16, -1), (22, -1), (9, -1), (2, -1), (32, -1), (24, -1), (6, -1), (11, -1), (26, -1), (28, -1), (5, -1), (14, -1), (20, -1), (21, -1), (7, -1)], 2: [(2, -1), (11, -1), (6, -1), (5, -1), (14, -1), (9, -1), (13, -1), (10, -1), (12, -1), (15, -1), (4, -1), (17, -1), (3, -1), (33, -1), (34, -1), (32, -1), (26, -1), (29, -1), (1, -1), (25, -1), (16, -1), (0, -1), (8, -1), (30, -1), (24, -1), (31, -1), (20, -1), (27, -1)], 3: [(3, -1), (4, -1), (17, -1), (5, -1), (11, -1), (33, -1), (2, -1), (30, -1), (9, -1), (34, -1), (6, -1), (14, -1), (1, -1), (16, -1), (29, -1), (26, -1), (12, -1), (20, -1), (8, -1), (13, -1), (0, -1), (10, -1), (7, -1), (23, -1), (15, -1), (27, -1), (24, -1), (22, -1), (32, -1), (25, -1), (19, -1), (18, -1)], 4: [(4, -1), (3, -1), (17, -1), (5, -1), (11, -1), (9, -1), (2, -1), (6, -1), (14, -1), (33, -1), (34, -1), (30, -1), (12, -1), (13, -1), (10, -1), (1, -1), (15, -1), (20, -1), (29, -1), (26, -1), (8, -1), (0, -1), (16, -1), (7, -1), (32, -1), (23, -1), (27, -1), (22, -1), (19, -1), (18, -1), (28, -1), (24, -1), (21, -1)], 5: [(5, -1), (6, -1), (11, -1), (2, -1), (14, -1), (9, -1), (13, -1), (12, -1), (10, -1), (15, -1), (17, -1), (4, -1), (3, -1), (33, -1), (34, -1), (32, -1), (26, -1), (29, -1), (16, -1), (8, -1), (1, -1), (25, -1), (0, -1), (20, -1), (7, -1), (31, -1), (24, -1), (30, -1), (19, -1)], 6: [(6, -1), (14, -1), (11, -1), (5, -1), (2, -1), (9, -1), (13, -1), (10, -1), (12, -1), (15, -1), (4, -1), (17, -1), (3, -1), (33, -1), (34, -1), (29, -1), (26, -1), (32, -1), (16, -1), (25, -1), (1, -1), (0, -1), (31, -1), (8, -1), (24, -1), (20, -1), (19, -1), (7, -1)], 7: [(7, -1), (23, -1), (20, -1), (17, -1), (25, -1), (4, -1), (33, -1), (34, -1), (29, -1), (3, -1), (30, -1), (26, -1), (18, -1), (27, -1), (16, -1), (5, -1), (22, -1), (28, -1), (1, -1), (11, -1), (21, -1), (15, -1), (12, -1), (24, -1), (14, -1), (19, -1), (6, -1), (10, -1)], 8: [(8, -1), (1, -1), (3, -1), (4, -1), (16, -1), (17, -1), (24, -1), (9, -1), (30, -1), (13, -1), (12, -1), (10, -1), (14, -1), (11, -1), (33, -1), (22, -1), (34, -1), (5, -1), (27, -1), (2, -1), (25, -1), (6, -1), (18, -1), (32, -1), (29, -1)], 9: [(9, -1), (14, -1), (6, -1), (11, -1), (2, -1), (13, -1), (5, -1), (10, -1), (12, -1), (15, -1), (4, -1), (3, -1), (17, -1), (33, -1), (34, -1), (29, -1), (30, -1), (26, -1), (8, -1), (32, -1), (1, -1), (22, -1), (25, -1), (31, -1), (27, -1), (18, -1), (0, -1), (23, -1), (24, -1), (16, -1)], 10: [(10, -1), (13, -1), (12, -1), (15, -1), (14, -1), (6, -1), (9, -1), (11, -1), (2, -1), (5, -1), (4, -1), (16, -1), (1, -1), (3, -1), (17, -1), (30, -1), (8, -1), (29, -1), (0, -1), (31, -1), (25, -1), (32, -1), (24, -1), (33, -1), (23, -1), (20, -1)], 11: [(11, -1), (2, -1), (6, -1), (5, -1), (14, -1), (9, -1), (13, -1), (12, -1), (10, -1), (15, -1), (4, -1), (17, -1), (3, -1), (33, -1), (34, -1), (32, -1), (26, -1), (29, -1), (1, -1), (8, -1), (25, -1), (0, -1), (31, -1), (24, -1), (20, -1), (30, -1), (19, -1), (7, -1), (21, -1)], 12: [(12, -1), (13, -1), (10, -1), (15, -1), (14, -1), (6, -1), (11, -1), (9, -1), (5, -1), (2, -1), (4, -1), (16, -1), (3, -1), (17, -1), (8, -1), (30, -1), (0, -1), (29, -1), (31, -1), (32, -1), (22, -1), (24, -1), (23, -1), (20, -1), (27, -1), (18, -1), (25, -1), (34, -1), (7, -1), (33, -1)], 13: [(13, -1), (10, -1), (12, -1), (15, -1), (14, -1), (6, -1), (11, -1), (9, -1), (2, -1), (5, -1), (4, -1), (16, -1), (3, -1), (17, -1), (1, -1), (30, -1), (8, -1), (0, -1), (29, -1), (31, -1), (32, -1), (25, -1), (24, -1), (33, -1), (23, -1), (28, -1)], 14: [(14, -1), (6, -1), (9, -1), (11, -1), (2, -1), (5, -1), (13, -1), (10, -1), (12, -1), (15, -1), (4, -1), (17, -1), (3, -1), (33, -1), (34, -1), (29, -1), (26, -1), (32, -1), (16, -1), (30, -1), (25, -1), (8, -1), (0, -1), (1, -1), (24, -1), (20, -1), (19, -1), (18, -1), (28, -1)], 15: [(15, -1), (10, -1), (13, -1), (12, -1), (9, -1), (14, -1), (6, -1), (2, -1), (11, -1), (5, -1), (4, -1), (34, -1), (33, -1), (30, -1), (16, -1), (0, -1), (32, -1), (31, -1), (26, -1), (29, -1), (3, -1), (22, -1), (23, -1), (20, -1), (27, -1), (18, -1), (25, -1), (7, -1), (8, -1)], 16: [(16, -1), (32, -1), (13, -1), (10, -1), (3, -1), (12, -1), (30, -1), (8, -1), (17, -1), (4, -1), (15, -1), (6, -1), (1, -1), (14, -1), (5, -1), (2, -1), (29, -1), (31, -1), (26, -1), (20, -1), (28, -1), (18, -1), (7, -1), (19, -1), (34, -1), (22, -1), (24, -1), (23, -1), (9, -1), (25, -1)], 17: [(17, -1), (3, -1), (4, -1), (5, -1), (11, -1), (6, -1), (33, -1), (2, -1), (34, -1), (9, -1), (14, -1), (30, -1), (29, -1), (20, -1), (26, -1), (1, -1), (7, -1), (12, -1), (23, -1), (8, -1), (13, -1), (16, -1), (0, -1), (10, -1), (32, -1), (19, -1), (24, -1), (25, -1), (22, -1), (27, -1), (21, -1), (31, -1)], 18: [(1, -1), (18, -1), (27, -1), (25, -1), (29, -1), (26, -1), (20, -1), (0, -1), (31, -1), (7, -1), (8, -1), (23, -1), (19, -1), (16, -1), (21, -1), (9, -1), (15, -1), (12, -1), (4, -1), (28, -1), (32, -1), (30, -1), (22, -1), (33, -1)], 19: [(19, -1), (21, -1), (25, -1), (17, -1), (29, -1), (26, -1), (33, -1), (4, -1), (18, -1), (27, -1), (22, -1), (16, -1), (3, -1), (32, -1), (34, -1), (28, -1), (11, -1), (7, -1), (6, -1), (5, -1), (14, -1), (20, -1), (9, -1)], 20: [(7, -1), (23, -1), (17, -1), (20, -1), (25, -1), (4, -1), (3, -1), (29, -1), (33, -1), (26, -1), (34, -1), (30, -1), (18, -1), (27, -1), (22, -1), (16, -1), (24, -1), (28, -1), (19, -1), (1, -1), (5, -1), (21, -1), (15, -1), (12, -1), (11, -1), (0, -1), (10, -1), (6, -1), (14, -1), (32, -1), (2, -1)], 21: [(21, -1), (19, -1), (25, -1), (29, -1), (26, -1), (18, -1), (20, -1), (33, -1), (7, -1), (27, -1), (1, -1), (34, -1), (28, -1), (17, -1), (4, -1), (31, -1), (24, -1)], 22: [(22, -1), (1, -1), (9, -1), (29, -1), (8, -1), (4, -1), (26, -1), (20, -1), (12, -1), (25, -1), (15, -1), (3, -1), (27, -1), (17, -1), (30, -1), (7, -1), (19, -1), (23, -1), (18, -1), (33, -1), (16, -1), (21, -1), (28, -1), (34, -1), (32, -1), (6, -1)], 23: [(7, -1), (23, -1), (20, -1), (17, -1), (25, -1), (29, -1), (4, -1), (33, -1), (34, -1), (26, -1), (30, -1), (3, -1), (15, -1), (18, -1), (27, -1), (12, -1), (10, -1), (22, -1), (28, -1), (13, -1), (16, -1), (32, -1), (9, -1)], 24: [(24, -1), (8, -1), (1, -1), (17, -1), (20, -1), (10, -1), (13, -1), (12, -1), (3, -1), (14, -1), (6, -1), (33, -1), (25, -1), (34, -1), (11, -1), (5, -1), (2, -1), (4, -1), (30, -1), (32, -1), (26, -1), (16, -1), (7, -1), (9, -1), (23, -1), (21, -1)], 25: [(25, -1), (29, -1), (26, -1), (20, -1), (1, -1), (18, -1), (27, -1), (7, -1), (19, -1), (23, -1), (33, -1), (21, -1), (28, -1), (34, -1), (6, -1), (14, -1), (22, -1), (9, -1), (10, -1), (2, -1), (11, -1), (5, -1), (17, -1), (0, -1), (13, -1), (8, -1), (30, -1), (31, -1), (3, -1), (24, -1), (12, -1), (15, -1), (32, -1)], 26: [(26, -1), (33, -1), (34, -1), (17, -1), (30, -1), (29, -1), (25, -1), (3, -1), (14, -1), (4, -1), (9, -1), (6, -1), (11, -1), (5, -1), (2, -1), (20, -1), (18, -1), (27, -1), (7, -1), (19, -1), (23, -1), (1, -1), (15, -1), (22, -1), (21, -1), (16, -1), (32, -1), (0, -1), (28, -1), (31, -1), (24, -1)], 27: [(27, -1), (1, -1), (18, -1), (25, -1), (22, -1), (0, -1), (29, -1), (26, -1), (20, -1), (4, -1), (8, -1), (3, -1), (7, -1), (32, -1), (23, -1), (9, -1), (19, -1), (17, -1), (15, -1), (12, -1), (30, -1), (21, -1)], 28: [(28, -1), (25, -1), (1, -1), (29, -1), (20, -1), (16, -1), (26, -1), (7, -1), (19, -1), (23, -1), (4, -1), (22, -1), (21, -1), (18, -1), (30, -1), (13, -1), (8, -1), (14, -1)], 29: [(17, -1), (25, -1), (26, -1), (33, -1), (3, -1), (34, -1), (9, -1), (14, -1), (4, -1), (30, -1), (6, -1), (29, -1), (20, -1), (11, -1), (5, -1), (1, -1), (2, -1), (18, -1), (27, -1), (10, -1), (7, -1), (19, -1), (23, -1), (13, -1), (22, -1), (12, -1), (21, -1), (28, -1), (15, -1), (16, -1), (8, -1), (0, -1), (31, -1), (24, -1)], 30: [(30, -1), (33, -1), (34, -1), (3, -1), (4, -1), (17, -1), (26, -1), (1, -1), (9, -1), (29, -1), (16, -1), (15, -1), (32, -1), (13, -1), (20, -1), (8, -1), (0, -1), (12, -1), (10, -1), (14, -1), (31, -1), (7, -1), (23, -1), (2, -1), (25, -1), (22, -1), (27, -1), (5, -1), (24, -1), (18, -1), (11, -1), (21, -1)], 31: [(31, -1), (1, -1), (15, -1), (30, -1), (10, -1), (18, -1), (13, -1), (12, -1), (9, -1), (32, -1), (16, -1), (6, -1), (33, -1), (25, -1), (34, -1), (11, -1), (5, -1), (17, -1), (29, -1), (2, -1), (26, -1), (20, -1), (21, -1), (4, -1)], 32: [(32, -1), (16, -1), (11, -1), (5, -1), (2, -1), (14, -1), (30, -1), (6, -1), (9, -1), (17, -1), (15, -1), (1, -1), (4, -1), (13, -1), (31, -1), (12, -1), (10, -1), (34, -1), (33, -1), (8, -1), (3, -1), (27, -1), (26, -1), (0, -1), (19, -1), (18, -1), (24, -1), (22, -1), (23, -1), (20, -1), (25, -1)], 33: [(33, -1), (34, -1), (30, -1), (17, -1), (3, -1), (4, -1), (26, -1), (9, -1), (14, -1), (2, -1), (6, -1), (11, -1), (29, -1), (5, -1), (15, -1), (20, -1), (1, -1), (25, -1), (7, -1), (23, -1), (8, -1), (19, -1), (0, -1), (32, -1), (31, -1), (10, -1), (24, -1), (21, -1), (22, -1), (13, -1), (12, -1)], 34: [(34, -1), (33, -1), (30, -1), (17, -1), (3, -1), (4, -1), (26, -1), (9, -1), (14, -1), (2, -1), (6, -1), (11, -1), (5, -1), (29, -1), (15, -1), (20, -1), (1, -1), (7, -1), (25, -1), (23, -1), (8, -1), (32, -1), (0, -1), (31, -1), (24, -1), (12, -1), (19, -1), (16, -1), (22, -1), (21, -1)]}

    evaluator._estimate_single_discovery_accuracy(frame_id=0, tau=1, pc_results=pc_results)