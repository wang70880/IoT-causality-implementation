import src.event_processing as evt_proc
import collections
import itertools
import numpy as np

from collections import defaultdict 

from src.event_processing import Hprocessor
from src.genetic_type import DevAttribute, AttrEvent, DataFrame
from src.tigramite.tigramite import data_processing as pp

class BackgroundGenerator():

    def __init__(self, dataset, event_processor, partition_config, tau_max) -> None:
        self.dataset = dataset
        self.partition_config = partition_config
        self.tau_max = tau_max
        self.event_processor:'Hprocessor' = event_processor

        self.temporal_pair_dict, self.heuristic_temporal_pair_dict = self._temporal_pair_identification()
        self.spatial_pair_dict = self._spatial_pair_identification()
        self.functionality_pair_dict = self._functionality_pair_identification()
        #self.candidate_pair_dict = self._candidate_pair_identification()

        self.correlation_dict = {
            'heuristic-temporal': self.heuristic_temporal_pair_dict,
            'temporal': self.temporal_pair_dict,
            'spatial': self.spatial_pair_dict,
            'functionality': self.functionality_pair_dict,
            #'candidate': self.candidate_pair_dict
        }
    
    def _temporal_pair_identification(self):
        """
        Many temporal pairs stem from the disorder of IoT logs. Therefore, we filter out pairs with low frequencies.
        The selected pairs are indexed by the time lag tau and the partitioning scheme (i.e., the date).

        Args:
            partition_config (int): _description_
            tau_max (int, optional): _description_. Defaults to 1.
        Returns:

        """
        # Return variables
        temporal_pair_dict = defaultdict(dict) # frame_id -> lag -> adjacency array of shape num_attrs X num_attrs
        heuristic_temporal_pair_dict = defaultdict(dict)

        # Auxillary variables
        name_device_dict:'dict[DevAttribute]' = self.event_processor.name_device_dict
        frame_dict = self.event_processor.frame_dict
        attr_names = self.event_processor.attr_names; num_attrs = len(attr_names)

        for frame_id in frame_dict.keys(): # Collect all lagged pairs in all frames
            frame: 'DataFrame' = frame_dict[frame_id]
            event_sequence:'list[AttrEvent]' = [tup[0] for tup in frame.training_events_states]
            for lag in range (1, self.tau_max + 1): # Initialize the count array
                temporal_pair_dict[frame_id][lag] = np.zeros(shape=(num_attrs, num_attrs), dtype=np.int32)
                heuristic_temporal_pair_dict[frame_id][lag] = np.zeros(shape=(num_attrs, num_attrs), dtype=np.int32)
            for i in range(len(event_sequence)): # Count the occurrence of each lagged aattr pair
                for lag in range (1, self.tau_max + 1):
                    if i + lag >= len(event_sequence):
                        continue
                    prior_attr_index = name_device_dict[event_sequence[i].dev].index; con_attr_index = name_device_dict[event_sequence[i + lag].dev].index
                    temporal_pair_dict[frame_id][lag][prior_attr_index, con_attr_index] += 1
                    heuristic_temporal_pair_dict[frame_id][lag][prior_attr_index, con_attr_index] += 1

        for frame_id in frame_dict.keys(): # JC NOTE: Here the frequency threshold is set empirically (Use partitioning criteria)
            for lag in range (1, self.tau_max + 1):
                count_array = temporal_pair_dict[frame_id][lag]
                heuristic_count_array = heuristic_temporal_pair_dict[frame_id][lag]
                for idx, x in np.ndenumerate(count_array):
                    heuristic_count_array[idx] = 0 if x < self.partition_config else 1 
                    count_array[idx] = 0 if x < self.partition_config else 1  

        return temporal_pair_dict, heuristic_temporal_pair_dict
    
    def _testbed_area_information(self):
        area_list = None
        area_connectivity_array = None
        if self.dataset == 'hh101':
            # In the list, each element (which is also a list) represents a physical area and deployed devices in that area.
            area_list = [['T102', 'D002', 'M001', 'LS001', 'M010', 'LS010'], \
                         ['M011', 'LS011', 'D003', 'T103', 'MA015', 'LS015'], \
                         ['M009', 'LS009', 'MA014', 'LS014', 'M012', 'LS012'], \
                         ['D001', 'T101', 'T104', 'T105', 'M005', 'LS005', 'MA013', 'LS013', 'M008', 'LS008', 'M004', 'LS004'], \
                         ['MA016', 'LS016', 'M003', 'LS003', 'M002', 'LS002', 'M007', 'LS007', 'M006', 'LS006']]
            num_areas = 5
            area_connectivity_array = np.zeros(shape=(num_areas, num_areas), dtype=np.int64)
            for i in range(num_areas):
                area_connectivity_array[i, i] = 1
            area_connectivity_array[0, 1] = 1; area_connectivity_array[1, 2] = 1; area_connectivity_array[0, 3] = 1; area_connectivity_array[3, 4] = 1
            area_connectivity_array[1, 0] = 1; area_connectivity_array[2, 1] = 1; area_connectivity_array[3, 0] = 1; area_connectivity_array[4, 3] = 1
        elif self.dataset == 'hh130':
            area_list = [['D002', 'T102', 'M001', 'LS001'], \
                         ['M002', 'LS002', 'T103', 'LS008'], \
                         ['M003', 'M004', 'LS003', 'LS004', 'T106'], \
                         ['M005', 'LS005', 'M006', 'LS006', 'LS009', 'LS010'], \
                         ['M011', 'T104', 'LS011', 'LS007']]
            num_areas = 5
            area_connectivity_array = np.zeros(shape=(num_areas, num_areas), dtype=np.int64)
            for i in range(num_areas):
                area_connectivity_array[i, i] = 1
            area_connectivity_array[0, 1] = 1; area_connectivity_array[0, 2] = 1; area_connectivity_array[1, 2] = 1; area_connectivity_array[2, 3] = 1; area_connectivity_array[3, 4] = 1
            area_connectivity_array[1, 0] = 1; area_connectivity_array[2, 0] = 1; area_connectivity_array[2, 1] = 1; area_connectivity_array[3, 2] = 1; area_connectivity_array[4, 3] = 1

        return area_list, area_connectivity_array

    def _spatial_pair_identification(self):
        # Return variables
        spatial_pair_dict = None
        # Auxillary variables
        area_list, area_connectivity_array = self._testbed_area_information()
        name_device_dict:'dict[DevAttribute]' = self.event_processor.name_device_dict
        frame_dict =self.event_processor.frame_dict
        attr_names = self.event_processor.attr_names; num_attrs = len(attr_names)

        if area_list:
            spatial_pair_dict = defaultdict(dict) # frame_id -> lag -> adjacency array of shape num_attrs X num_attrs
            spatial_array = np.zeros(shape=(num_attrs, num_attrs), dtype=np.int32)
            for lag in range (1, self.tau_max + 1): # The spatial array is only concerned with the lag and the testbed_area_list
                #transition_area_array_for_cur_lag = area_connectivity_array if lag == 1 else np.linalg.matrix_power(area_connectivity_array, lag)
                #transition_area_array_for_cur_lag = area_connectivity_array
                for index, x in np.ndenumerate(area_connectivity_array):
                    if x == 0:
                        continue
                    pre_area = area_list[index[0]]; con_area = area_list[index[1]]
                    for element in itertools.product(pre_area, con_area): # Get the cartesian product for devices in these two areas
                        if all([element[i] in attr_names for i in range(2)]):
                            spatial_array[name_device_dict[element[0]].index, name_device_dict[element[1]].index] = 1

            for frame_id in frame_dict.keys():
                for lag in range (1, self.tau_max + 1): 
                    spatial_pair_dict[frame_id][lag] = spatial_array
        return spatial_pair_dict 

    def _functionality_pair_identification(self):

        # Return variables
        functionality_pair_dict = {} # First index: keyword in {'activity', 'physics'}. Value: an integer array of shape num_attrs * num_attrs

        # Auxillary variables
        name_device_dict:'dict[DevAttribute]' = self.event_processor.name_device_dict
        attr_names = self.event_processor.attr_names; num_attrs = len(attr_names)

        functionality_pair_dict['activity'] = np.zeros(shape=(num_attrs, num_attrs), dtype=np.int64)
        functionality_pair_dict['physics'] = np.zeros(shape=(num_attrs, num_attrs), dtype=np.int64)
        for i in range(num_attrs):
            for j in range(num_attrs):
                activity_flags = [attr_names[k].startswith(('M', 'D')) for k in [i, j]] # Identify attributes which are related with user activities 
                physics_flags = [attr_names[k].startswith(('M', 'T', 'LS')) for k in [i, j]] # Identify attributes which are related with physics
                functionality_pair_dict['activity'][i, j] = 1 if all(activity_flags) else 0
                functionality_pair_dict['physics'][i, j] = 1 if all(physics_flags) else 0

        return functionality_pair_dict

    def _candidate_pair_identification(self):
        
        # Return variables
        candidate_pair_dict = {}

        # Auxillary variables
        frame_dict = self.event_processor.frame_dict
        name_device_dict:'dict[DevAttribute]' = self.event_processor.name_device_dict
        attr_names = self.event_processor.attr_names; num_attrs = len(attr_names)

        merged_functionality_pair_dict = self.functionality_pair_dict['activity'] + self.functionality_pair_dict['physics']
        merged_functionality_pair_dict[merged_functionality_pair_dict >= 1] = 1
        for frame_id in frame_dict.keys():
            candidate_pair_dict[frame_id] = {}
            for lag in range (1, self.tau_max + 1):
                candidate_pair_dict[frame_id][lag] = self.temporal_pair_dict[frame_id][lag] + self.spatial_pair_dict[frame_id][lag]\
                                                    + merged_functionality_pair_dict 
                candidate_pair_dict[frame_id][lag][candidate_pair_dict[frame_id][lag] < 3] = 0
                candidate_pair_dict[frame_id][lag][candidate_pair_dict[frame_id][lag] == 3] = 1
        return candidate_pair_dict

    def _print_pair_list(self, interested_array):
        attr_names = self.event_processor.attr_names; num_attrs = len(attr_names)
        pair_list = []
        for index, x in np.ndenumerate(interested_array):
            if x == 1:
                pair_list.append((attr_names[index[0]], attr_names[index[1]]))
        print("Pair list with lens {}: {}".format(len(pair_list), pair_list))

    def print_benchmark_info(self,frame_id=0, tau=1, type = ''):
        """Print out the identified device correlations.

        Args:
            frame_id (int, optional): _description_. Defaults to 0.
            tau (int, optional): _description_. Defaults to 1.
            type (str, optional): 'temporal' or 'spatial' or 'functionality'
        """
        print("The {} corrleation dict for frame_id = {}, tau = {}: ".format(type, frame_id, tau))
        attr_names = self.event_processor.attr_names; num_attrs = len(attr_names)
        if type != 'functionality':
            self._print_pair_list(self.correlation_dict[type][frame_id][tau])
        else:
            self._print_pair_list(self.correlation_dict[type]['activity'])
            self._print_pair_list(self.correlation_dict[type]['physics'])

    def apply_background_knowledge(self, selected_links=None, knowledge_type='', frame_id=0):
        assert(selected_links is not None)
        n_filtered_edges = 0; n_qualified_edges = 0
        for worker_index, link_dict in selected_links.items():
            for outcome, cause_list in link_dict.items():
                new_cause_list = []
                for (cause, lag) in cause_list:
                    background_array = self.correlation_dict[knowledge_type][frame_id][abs(lag)] \
                        if knowledge_type != 'functionality' else self.correlation_dict[knowledge_type]['activity'] + self.correlation_dict[knowledge_type]['physics']
                    background_array[background_array >= 1] = 1 # Normalize the background array
                    if background_array[cause, outcome] > 0:
                        new_cause_list.append((cause, lag))
                        n_qualified_edges += 1
                    else:
                        n_filtered_edges += 1
                selected_links[worker_index][outcome] = new_cause_list
        print("[Background Generator] By applying {} knowledge, CausalIoT filtered {} edges.".format(knowledge_type, n_filtered_edges))
        print("[Background Generator] # of candidate edges: {}.".format(n_qualified_edges))
        return selected_links

    def generate_candidate_interactions(self, apply_bk, frame_id, N, autocorrelation_flag=True):
        if autocorrelation_flag:
            selected_links = {n: {m: [(i, -t) for i in range(N) for \
                t in range(1, self.tau_max + 1)] if m == n else [] for m in range(N)} for n in range(N)}
        else:
            selected_links = {n: {m: [(i, -t) for i in range(N) if i != m for \
                t in range(1, self.tau_max + 1)] if m == n else [] for m in range(N)} for n in range(N)}
        if apply_bk == 0:
            pass
        if apply_bk >= 1:
            selected_links = self.apply_background_knowledge(selected_links, 'heuristic-temporal', frame_id)
        if apply_bk >= 2:
            selected_links = self.apply_background_knowledge(selected_links, 'spatial', frame_id)
            selected_links = self.apply_background_knowledge(selected_links, 'functionality', frame_id)
        return selected_links

if __name__ == '__main__':
    # Parameter setting
    dataset = 'hh101'
    partition_config = 10
    tau_max = 2; tau_min = 1
    event_preprocessor = evt_proc.Hprocessor(dataset)
    attr_names, dataframes = event_preprocessor.initiate_data_preprocessing(partition_config=partition_config)
    background_generator = BackgroundGenerator(dataset, event_preprocessor, partition_config, tau_max)
    background_generator.print_benchmark_info(frame_id=3, tau=2, type='spatial')
