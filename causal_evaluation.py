from src.event_processing import Hprocessor
import collections
import itertools
import numpy as np
class Evaluator():

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.event_processor = Hprocessor(dataset)
        self.temporal_pair_dict = None
        self.spatial_array = None
    
    def _temporal_pair_identification(self, partition_config, tau_max=1):
        """Identification of the temporal correlation pair in the following format.

            (attr_c, attr_o, E[attr_o|attr_c=0], E[attr_o|attr_c=1])

            These pairs are indexed by the time lag tau and the partitioning scheme (i.e., the date).

        Args:
            partition_config (tuple): _description_
            tau_max (int, optional): _description_. Defaults to 1.
        Returns:

        """
        temporal_pair_dict = {} # First index: frame_id. Second index: lag. Value: an integer array of shape num_attrs X num_attrs
        self.event_processor.initiate_data_preprocessing(partition_config=partition_config)
        for frame_id in range( len(self.event_processor.frame_dict.keys())): # Collect all lagged pairs in all frames
            temporal_pair_dict[frame_id] = {}
            attr_names = self.event_processor.attr_names; num_attrs = len(attr_names)
            attr_sequence = self.event_processor.frame_dict[frame_id]['attr-sequence']
            state_sequence = self.event_processor.frame_dict[frame_id]['state-sequence']
            assert(len(attr_sequence) == len(state_sequence))
            num_event = len(attr_sequence)
            for event_id in range(num_event): # Count the occurrence of each attr pair and update the corresponding array
                for lag in range (1, tau_max + 1):
                    if event_id + lag >= num_event:
                        continue
                    temporal_pair_dict[frame_id][lag] = np.zeros(shape=(num_attrs, num_attrs), dtype=np.int64) if lag not in temporal_pair_dict[frame_id].keys() else temporal_pair_dict[frame_id][lag]
                    prior_attr = attr_sequence[event_id]; con_attr = attr_sequence[event_id + lag]
                    temporal_pair_dict[frame_id][lag][attr_names.index(prior_attr), attr_names.index(con_attr)] += 1

        for frame_id in range( len(self.event_processor.frame_dict.keys())):
            for lag in range (1, tau_max + 1):
                attr_array = temporal_pair_dict[frame_id][lag]
                pair_sum = np.sum(attr_array)
                for idx, x in np.ndenumerate(attr_array):
                    attr_array[idx] = 0 if x < pair_sum * 0.001 else 1 # NOTE: Filter out pairs with low frequencies (threshold: %1 of the total frequency)

        # print("[frame_id=0, lag=1] Number of temporal pairs is{}".format(np.sum(temporal_pair_dict[0][1])))
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
                

    def physical_correlation_identification():
        pass

    def automation_correlation_identification():
        pass

if __name__ == '__main__':
    evaluator = Evaluator('hh101')
    partition_config = (1, 10)
    tau_max = 1
    evaluator._temporal_pair_identification(partition_config=partition_config, tau_max=tau_max)
    evaluator._spatial_correlation_identification()